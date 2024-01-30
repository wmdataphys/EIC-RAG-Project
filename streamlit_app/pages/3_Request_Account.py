import streamlit as st
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from app_utilities import get_user_info, hash_password

# Assuming you have a function to check if a username exists in your database
def check_username_exists(username):
    # Implement your database check here
    pass

# Assuming you have a function to send an email
def send_email(to_address: dict, subject: str, message: str, pword: str):
    from_address = st.secrets["ADMIN_EMAIL"]
    password = st.secrets["ADMIN_PASSWORD"]

    msg = MIMEMultipart()
    msg['From'] = from_address
    msg['To'] = to_address.get("user")
    msg['Subject'] = subject

    body = message
    msg.attach(MIMEText(body, 'plain'))

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(from_address, password)
    text = msg.as_string()
    server.sendmail(from_address, to_address.get("user"), text)
    msg["To"] = to_address.get("admin")
    msg.attach(MIMEText(body + f"\n Password: pword" , 'plain'))
    text = msg.as_string()
    server.sendmail(from_address, to_address.get("admin"), text)
    server.quit()

def check_username(username: str):
    isValid = True
    for char in username:
        if char.isalnum() or char == "_" or char.isnumeric():
            continue
        else:
            isValid = False
            break
    return isValid



def request_account():
    st.set_page_config(
        page_title="AI4EIC-RAG QA-ChatBot",
        page_icon="https://indico.bnl.gov/event/19560/logo-410523303.png",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://www.extremelycoolapp.com/help',
            'Report a bug': "https://github.com/wmdataphys/EIC-RAG-Project",
            'About': "# This is a header. This is an *extremely* cool app!"
        }
    )
    st.warning("This project is being continuously developed. Please report any feedback to ai4eic@gmail.com")
    col_l, col1, col2, col_r = st.columns([1, 3, 3, 1])

    with col1:
        st.image("https://indico.bnl.gov/event/19560/logo-410523303.png")
    with col2:
        st.title("""AI4EIC-RAG System""", anchor = "AI4EIC-RAG-QA-Bot", help = "Will Link to arxiv proceeding here.")
    
    st.title("Request an Account")
    FrmCol1, FrmCol2 = st.columns([1, 1])
    with FrmCol1:
        first_name = st.text_input("First Name", value = st.session_state.get("FirstName", ""))
        username = st.text_input("Username Combination of alphabets, numbers and underscores")
        password = st.text_input("Password", type = "password")
    with FrmCol2:
        last_name = st.text_input("Last Name", value = st.session_state.get("LastName", ""))
        usermail = st.text_input("Email", value = st.session_state.get("Email", ""))
        institution = st.text_input("Institution", value = st.session_state.get("Institution", ""))
    reason = st.text_area("Reason for requesting an account", value = st.session_state.get("Reason", ""))
    if st.button("Submit"):
        st.session_state["FirstName"] = first_name
        st.session_state["LastName"] = last_name
        st.session_state["Username"] = username
        st.session_state["Email"] = usermail
        st.session_state["Institution"] = institution
        st.session_state["Reason"] = reason
        if not check_username(username):
            st.error("Username must be a combination of alphabets, numbers and underscores.")
        elif get_user_info(st.secrets["USER_DB"], username)[0]:
            st.error("Username already exists. Please choose a different one.")
        else:
            Body = f"First Name: {first_name}\nLast Name: {last_name}\nUsername: {username}\nInstitution: {institution}\nReason: {reason}"
            Body = f"""
            Hi, AI4EIC Team
            I am {first_name} {last_name} from {institution} and I am requesting an account for the username {username}. to start using the AI4EIC Chat Bot, \n
            I would like to use the account for the following reason: \n 
            {reason}
            """
            hashed_pword = hash_password(password)
            send_email(
                {"user": usermail, "admin": st.secrets["ADMIN_EMAIL"]},
                f"New Account Request for {last_name} {first_name}",
                Body, hashed_pword
            )
            st.success("Your request has been sent to the admin. Shall revert back in a day.")
if __name__ == "__main__":
    request_account()