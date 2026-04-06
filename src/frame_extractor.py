import cv2
import yt_dlp
import base64
import os
import tempfile
import whisper_timestamped as whisper
from youtube_transcript_api import YouTubeTranscriptApi
from threading import Thread

class FrameExtractor:

    def __init__(self):

        self.frames = None
        self.transcript = None

        print("Loading Whisper model")
        self.whisper_model = whisper.load_model("tiny")

    def extract_frames(
        self,
        video_path,
        timestamp,
        context_seconds=3,
        fps_sample=2
    ):

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError("Could not open video")

        video_fps = cap.get(cv2.CAP_PROP_FPS)

        frames_to_extract = []

        # Add context frames
        for i in range(int(context_seconds * fps_sample) + 1):

            ts = timestamp - (i / fps_sample)

            if ts >= 0:
                frames_to_extract.append(ts)

        frames_to_extract = sorted(set(frames_to_extract))

        extracted_frames = []

        for ts in frames_to_extract:

            frame_idx = int(ts * video_fps)

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            ret, frame = cap.read()

            if ret:
                extracted_frames.append(frame)

        cap.release()

        return extracted_frames

    def encode_frames_to_base64(self, frames):

        base64_frames = []

        for frame in frames:

            _, buffer = cv2.imencode('.jpg', frame)

            jpg_as_text = base64.b64encode(
                buffer
            ).decode('utf-8')

            base64_frames.append(jpg_as_text)

        return base64_frames

    def get_transcript_from_youtube(
        self,
        video_id,
        start_time,
        end_time
    ):

        
        try:

            transcript = YouTubeTranscriptApi.get_transcript(
                video_id
            )

            words = []

            for segment in transcript:

                if (
                    segment["start"] < end_time
                    and
                    segment["start"] + segment["duration"]
                    > start_time
                ):

                    words.append(segment["text"])

            return " ".join(words)

        except Exception:

            return None
        
    def extract_transcript_whisper(
        self,
        video_path,
        start_time,
        end_time
    ):
        
        result = whisper.transcribe(
            self.whisper_model,
            video_path,
            language="en"
        )

        transcript_segment = []

        for segment in result['segments']:

            if (
                segment['start'] < end_time
                and
                segment['end'] > start_time
            ):

                transcript_segment.append(
                    segment['text']
                )

        return " ".join(transcript_segment)
        
    def build_video_context(
        self,
        youtube_url,
        timestamp,
        context_seconds=6,
        fps_sample=2
    ):

        video_context = {
            "frames": [],
            "transcript": "",
            "timestamp": timestamp
        }

        video_id = youtube_url.split("v=")[-1].split("/")[-1]

        with tempfile.TemporaryDirectory() as tmpdir:

            video_path = os.path.join(
                tmpdir,
                "video.mp4"
            )


            print("Downloading low-res video")

            ydl_opts = {

                'format': 'worstvideo[ext=mp4]+bestaudio[ext=m4a]/worst[ext=mp4]',

                'outtmpl': video_path,

                'quiet': True,

                'noplaylist': True,

            }

            try:

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:

                    ydl.download([youtube_url])

            except Exception as e:

                print("Download failed:", e)

                return None


            frames = self.extract_frames(
                video_path,
                timestamp,
                context_seconds,
                fps_sample
            )
            
            video_context["frames"] = self.encode_frames_to_base64(frames = frames)

            # transcript_start = max(
            #     0,
            #     timestamp - context_seconds
            # )

            transcript_start = 0

            transcript_end = timestamp + 2

            transcript = self.get_transcript_from_youtube(
                    video_id,
                    transcript_start,
                    transcript_end
                )

            
            if transcript is None:

                print(
                    "No captions — using Whisper fallback"
                )

                transcript = self.extract_transcript_whisper(
                        video_path,
                        transcript_start,
                        transcript_end
                    )

            video_context["transcript"] = transcript

            

        return video_context

if __name__ == '__main__':
    youtube_url = "https://youtu.be/nVyD6THcvDQ"  # Example URL
    timestamp = 60.0  # Example timestamp in seconds

    fe = FrameExtractor()
    print(f"Starting context extraction for {youtube_url} at {timestamp}s")
    context = fe.build_video_context(youtube_url, timestamp, context_seconds=5, fps_sample=1)

    if context:
        print("\n--- Extracted Video Context ---")
        print(f"Timestamp: {context['timestamp']}s")
        print(f"Number of frames (base64 encoded): {len(context['frames'])}")
        if len(context['frames']) > 0:
            print(f"First frame (base64, truncated): {context['frames'][0][:100]}...")
        print(f"Transcript segment: {context['transcript']}")
    else:
        print("Failed to build video context.")