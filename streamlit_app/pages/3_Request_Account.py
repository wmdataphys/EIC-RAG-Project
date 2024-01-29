import streamlit as st
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from app_utilities import get_user_info

# Assuming you have a function to check if a username exists in your database
def check_username_exists(username):
    # Implement your database check here
    pass

# Assuming you have a function to send an email
def send_email(to_address, subject, message):
    from_address = st.secrets["ADMIN_EMAIL"]
    password = st.secrets["ADMIN_PASSWORD"]

    msg = MIMEMultipart()
    msg['From'] = from_address
    msg['To'] = to_address
    msg['Subject'] = subject

    body = message
    msg.attach(MIMEText(body, 'plain'))

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(from_address, password)
    text = msg.as_string()
    server.sendmail(from_address, to_address, text)
    server.quit()

def request_account():
    st.title("Request an Account")

    first_name = st.text_input("First Name")
    last_name = st.text_input("Last Name")
    username = st.text_input("Username")

    if st.button("Submit"):
        if check_username_exists(username):
            st.error("Username already exists. Please choose a different one.")
        else:
            send_email(
                st.secrets["ADMIN_EMAIL"],
                "New Account Request",
                f"First Name: {first_name}\nLast Name: {last_name}\nUsername: {username}"
            )
            st.success("Your request has been sent to the admin.")
            

if __name__ == "__main__":
    request_account()