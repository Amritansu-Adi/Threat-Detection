import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email_alert(subject, body, to_email,
                     from_email='amritansuaditya1@gmail.com',
                     app_password='rnumfwbpvwwwigfd',
                     timeout: int = 10):
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    server = None
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587, timeout=timeout)
        server.ehlo()
        server.starttls()
        server.login(from_email, app_password)
        text = msg.as_string()
        server.sendmail(from_email, to_email, text)
        print('Email alert sent!')
    except Exception as e:
        print(f'Failed to send email: {e}')
    finally:
        # Ensure the SMTP connection is closed without blocking the main loop
        try:
            if server is not None:
                server.quit()
        except Exception:
            try:
                if server is not None:
                    server.close()
            except Exception:
                pass
