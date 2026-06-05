from enum import Enum

class CallSentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    MIXED = "mixed"

class SupportedAudioFormats(str, Enum):
    MP3 = "audio/mpeg"
    WAV = "audio/wav"
    M4A = "audio/mp4"
    WEBM = "audio/webm"
    MP4 = "video/mp4" # Sometimes audio is sent as video/mp4 format container
