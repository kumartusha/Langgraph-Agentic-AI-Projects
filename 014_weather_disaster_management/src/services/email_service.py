from src.config.settings import settings
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class EmailService:
    @staticmethod
    def send_alert(city: str, severity: str, disaster_type: str, weather_data: dict, response_plan: str, is_human_verified: bool = False):
        """
        Sends an email alert. 
        Note: Currently mocks the SMTP server call to prevent application crashes 
        if valid credentials are not provided in the environment.
        """
        sender_email = settings.SENDER_EMAIL
        receiver_email = settings.RECEIVER_EMAIL
        
        email_content = f"""
Weather Alert for {city}

Disaster Type: {disaster_type}
Severity Level: {severity}

Current Weather Conditions:
- Weather Description: {weather_data.get('weather')}
- Temperature: {weather_data.get('temperature')}C
- Wind Speed: {weather_data.get('wind_speed')} m/s
- Humidity: {weather_data.get('humidity')}%
- Pressure: {weather_data.get('pressure')} hPa
- Cloud Cover: {weather_data.get('cloud_cover')}%

Response Plan:
{response_plan}

This is an automated weather alert generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        if is_human_verified:
            email_content += "\nNote: This low/medium severity alert has been verified by a human operator."

        # Mocking SMTP for safety in this refactored environment
        print("="*50)
        print("MOCK EMAIL SENT:")
        print(f"From: {sender_email}")
        print(f"To: {receiver_email}")
        print(f"Subject: Weather Alert: {severity} severity weather event in {city}")
        print("-" * 50)
        print(email_content)
        print("="*50)
        
        logger.info(f"Mock email alert sent successfully for {city}")
        return True
