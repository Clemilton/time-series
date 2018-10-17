# import necessary packages
 
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib

def sendemail(to_addr,subject, message):


    # create message object instance
    msg = MIMEMultipart()
     
     
    message = message
    with open('login_email') as f:
        lines = f.readlines()

    # setup the parameters of the message
    msg['From'] = lines[0][:-1]
    password = lines[1]
    msg['To'] = to_addr
    msg['Subject'] = subject
     
    # add in the message body
    msg.attach(MIMEText(message, 'plain'))
     
    #create server
    server = smtplib.SMTP('smtp.gmail.com: 587')
     
    server.starttls()
     
    # Login Credentials for sending the mail
    server.login(msg['From'], password)
     
     
    # send the message via the server.
    server.sendmail(msg['From'], msg['To'], msg.as_string())
     
    server.quit()
     
    print ("successfully sent email to %s:" % (msg['To']))