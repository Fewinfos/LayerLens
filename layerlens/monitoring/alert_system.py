"""
Module for sending alerts when issues are detected in monitored models.
"""

import logging
import smtplib
import datetime
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class AlertSystem:
    """System for sending alerts (Slack, Email, etc.)."""
    
    def __init__(self, config=None):
        """
        Initialize the alert system.
        
        Args:
            config (dict): Configuration for the alert system
        """
        self.config = config or {}
        self.alert_history = []
        self.alert_count = 0
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('layerlens.alert_system')
    
    def send_alert(self, title, message, severity='warning', channels=None):
        """
        Send an alert through configured channels.
        
        Args:
            title (str): Alert title
            message (str): Alert message
            severity (str): Alert severity ('info', 'warning', 'error', 'critical')
            channels (list): List of channels to send to (if None, uses all configured)
            
        Returns:
            bool: Whether the alert was sent successfully
        """
        # Create the alert
        alert = {
            'title': title,
            'message': message,
            'severity': severity,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Determine which channels to use
        if channels is None:
            channels = self.config.get('channels', ['log'])
        
        # Send to each channel
        success = True
        for channel in channels:
            channel_success = self._send_to_channel(channel, alert)
            success = success and channel_success
        
        # Record the alert
        self.alert_history.append(alert)
        self.alert_count += 1
        
        return success
    
    def _send_to_channel(self, channel, alert):
        """
        Send an alert to a specific channel.
        
        Args:
            channel (str): Channel to send to ('email', 'slack', 'log')
            alert (dict): Alert to send
            
        Returns:
            bool: Whether the alert was sent successfully
        """
        if channel == 'email':
            return self._send_email(alert)
        elif channel == 'slack':
            return self._send_slack(alert)
        elif channel == 'log':
            return self._send_log(alert)
        else:
            self.logger.warning(f"Unknown alert channel: {channel}")
            return False
    
    def _send_email(self, alert):
        """Send an alert via email."""
        email_config = self.config.get('email', {})
        if not email_config:
            self.logger.warning("Email channel configured but no email settings found")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = email_config.get('from', 'alerts@layerlens.ai')
            msg['To'] = email_config.get('to', 'admin@example.com')
            msg['Subject'] = f"[{alert['severity'].upper()}] {alert['title']}"
            
            # Add message body
            body = f"{alert['message']}\n\nTimestamp: {alert['timestamp']}"
            msg.attach(MIMEText(body, 'plain'))
            
            # Connect to SMTP server
            server = smtplib.SMTP(
                email_config.get('smtp_server', 'smtp.gmail.com'),
                email_config.get('smtp_port', 587)
            )
            server.starttls()
            
            # Login and send
            server.login(
                email_config.get('username', ''),
                email_config.get('password', '')
            )
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Email alert sent: {alert['title']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
            return False
    
    def _send_slack(self, alert):
        """Send an alert via Slack."""
        slack_config = self.config.get('slack', {})
        if not slack_config:
            self.logger.warning("Slack channel configured but no Slack settings found")
            return False
        
        try:
            import requests
            
            # Create payload
            payload = {
                'text': f"*[{alert['severity'].upper()}] {alert['title']}*\n{alert['message']}",
                'username': slack_config.get('username', 'LayerLens Alert'),
                'icon_emoji': slack_config.get('icon_emoji', ':robot_face:')
            }
            
            # Send request
            webhook_url = slack_config.get('webhook_url', '')
            if not webhook_url:
                self.logger.warning("No Slack webhook URL configured")
                return False
                
            response = requests.post(
                webhook_url,
                data=json.dumps(payload),
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                self.logger.info(f"Slack alert sent: {alert['title']}")
                return True
            else:
                self.logger.warning(f"Failed to send Slack alert: {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to send Slack alert: {e}")
            return False
    
    def _send_log(self, alert):
        """Send an alert to logs."""
        log_level = self._severity_to_log_level(alert['severity'])
        self.logger.log(
            log_level,
            f"ALERT - {alert['title']}: {alert['message']}"
        )
        return True
    
    def _severity_to_log_level(self, severity):
        """Convert alert severity to log level."""
        levels = {
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR,
            'critical': logging.CRITICAL
        }
        return levels.get(severity.lower(), logging.WARNING)
    
    def get_alert_history(self, start_time=None, end_time=None, severity=None):
        """
        Get alert history filtered by time and severity.
        
        Args:
            start_time: Start time for filtering (datetime or ISO format string)
            end_time: End time for filtering (datetime or ISO format string)
            severity: Severity to filter by
            
        Returns:
            List of alerts matching the criteria
        """
        # Convert string times to datetime if needed
        if isinstance(start_time, str):
            start_time = datetime.datetime.fromisoformat(start_time)
        if isinstance(end_time, str):
            end_time = datetime.datetime.fromisoformat(end_time)
        
        # Filter alerts
        filtered_alerts = []
        for alert in self.alert_history:
            # Parse timestamp
            if isinstance(alert['timestamp'], str):
                alert_time = datetime.datetime.fromisoformat(alert['timestamp'])
            else:
                alert_time = alert['timestamp']
            
            # Apply filters
            if start_time and alert_time < start_time:
                continue
            if end_time and alert_time > end_time:
                continue
            if severity and alert['severity'] != severity:
                continue
            
            filtered_alerts.append(alert)
        
        return filtered_alerts
    
    def configure(self, config):
        """
        Update alert system configuration.
        
        Args:
            config (dict): New configuration
        """
        self.config = config
        self.logger.info("Alert system configuration updated")
