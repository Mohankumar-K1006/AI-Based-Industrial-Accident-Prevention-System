# test_smtp_raw.py — bare minimum SMTP test, no try/except, no threading
import smtplib
import os
from email.mime.text import MIMEText

msg = MIMEText("This is a raw SMTP test from the industrial safety system.")
msg["Subject"] = "raw smtp test"
msg["From"] = os.environ["SMTP_USER"]
msg["To"] = os.environ["SMTP_TEST_TO"]

print("connecting...")
server = smtplib.SMTP(os.environ.get("SMTP_HOST", "smtp.gmail.com"),
                      int(os.environ.get("SMTP_PORT", "587")))
server.set_debuglevel(1)
server.starttls()
print("logging in...")
server.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"])
print("sending...")
server.sendmail(msg["From"], [msg["To"]], msg.as_string())
server.quit()
print("done")
