import tkinter as tk
from tkinter import filedialog,Text,Label
import tkinter.font as tkFont
from PIL import ImageTk,Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pygame


pygame.mixer.init()
new_model = tf.keras.models.load_model('saved_model/my_model')


batch_size = 32
img_height = 300
img_width = 300
class_names = ['broadbill', 'eagle', 'owl', 'parrot','woodpecker']
root = tk.Tk()
root.title('PROJECT BIRDS RECOGNITION!')
root.geometry("800x600")
filename ="null"
apps = []



def add_App():
    global filename, Bird_image
    
    # Prompt the user to select a file
    filename = filedialog.askopenfilename(initialdir="/", title="Select File",
                                          filetypes=(("all files", "*.*"), ("exe", "*.exe")))
    apps.append(filename)
    
    # Clear the previous image label
    labelproject.configure(image=None)
    
    # Display the selected image
    img = Image.open(filename)
    img = img.resize((300, 250))
    Bird_image = ImageTk.PhotoImage(img)
    labelproject.configure(image=Bird_image)
    
    return filename

#RUN APP button fuction
def run_App():
    global filename
    global labeloutput
    img = keras.preprocessing.image.load_img(
    filename, target_size=(img_height, img_width))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = new_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    Output = (
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score)))
    labeloutput = tk.Label(text=Output,bg="white")
    labeloutput.pack()

#CLEAR  APP button fuction   
def clear_App():
    global labeloutput
    if labeloutput:
        labeloutput.pack_forget()  # Remove the label from the GUI
        labeloutput = None  # Reset the labeloutput variable
    apps.clear()   
        
background_image = ImageTk.PhotoImage(Image.open("background_image.png"))
background_label = tk.Label(root, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)


Bird_image =ImageTk.PhotoImage(Image.open('bird.png'))

fontStyle = tkFont.Font(family="Poppins",size=20,weight="bold")
labeltitle = Label(text="PROJECT BIRD DETECTION!!!!!" ,font= fontStyle)
labeltitle.pack()

labelproject = Label(image=Bird_image)
labelproject.pack()

# line = tk.Frame( height=5, width=550, bg="grey80", relief='groove')
# line.pack()

#OPEN FILE button
openFile = tk.Button(text="Open File",padx=10,pady=5,fg="white",bg="#808080",  command=add_App  )
openFile.pack()
openFile.place(x=270, y=370)

#RUN APP button
RunApp = tk.Button(text="Run App ",padx=10,pady=5,fg="white",bg="#808080",  command=run_App  )
RunApp.pack()
# RunApp.pack(side='right')
RunApp.place(x=450, y=370)

#CLEAR APP button
clearApp = tk.Button(text="clear",padx=10,pady=5,fg="white",bg="#808080",  command=clear_App  )
clearApp.pack()
clearApp.place(x=370, y=370)


root.mainloop()