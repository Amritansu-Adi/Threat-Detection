import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

def send_email_alert(subject, body, image_path, to_email,
                     from_email='amritansuaditya1@gmail.com',
                     app_password='rnumfwbpvwwwigfd',
                     timeout: int = 10):
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # Attach the image
    if image_path:
        try:
            with open(image_path, 'rb') as f:
                img_data = f.read()
            image = MIMEImage(img_data, name=os.path.basename(image_path))
            msg.attach(image)
        except Exception as e:
            print(f"Failed to attach image to email: {e}")
            # Still try to send the email without the image
            pass

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
