from tkinter import *
import tkinter.font as font
from PIL import Image, ImageTk
import numpy as np
import cv2 as cv
import os
from prep_and_predict import predict, prep_img
import matplotlib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # hides tensorflow warnings

'''
Frontend: tkinter canvas where user can see what they're drawing, displays predictions
Backend: where the user draws on the canvas is also drawn on a cv2 image (both same size), which is a numpy array. When the user 
         clicks the predict button, that numpy array is prepped and fed into the neural net model. The predictions are displayed 
         on the tkinter canvas which the user can see
'''

CLASSES = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt' # 47 classes

FRAME_WIDTH = 800
FRAME_HEIGHT = 400

tk = Tk()
tk.title = 'Drawing w/ Mouse'
tk.geometry('{}x{}'.format(FRAME_WIDTH, FRAME_HEIGHT))
tk.resizable(False, False) # prevents changing window dimensions

rects_and_labels = []

def paint(event):
    color = 'black'
    x1,y1=(event.x-3), (event.y-3)
    x2,y2=(event.x+3), (event.y+3)
    canvas.create_oval(x1, y1, x2, y2, fill=color, outline=color) # small dot
    cv.circle(blank, (event.x, event.y), 3, (0, 0, 0), -1) # small dot also made in cv image

def predict_drawing():
    global rects_and_labels

    # clear any existing rectangles and labels drawn while keeping drawings
    for x in rects_and_labels:
        canvas.delete(x)

    rects_and_labels = []
    chars, chars_dimensions = prep_img(blank)
    predictions = predict(blank, chars, chars_dimensions)

    guessed_chars = []
    for i, p in enumerate(predictions): # i is counter, p is prediction list
        p_index = np.argmax(p)
        prediction = CLASSES[int(p_index)] # 0, 1, 2... x, y, z
        guessed_chars.append(str(prediction))
        rounded = round(p[p_index] * 100, 2)
        print(f'Prediction: {prediction} ({rounded}%)')

        # drawing rectangles around each char and displaying their prediction
        x, y, w, h = chars_dimensions[i]
        x1 = x - 4
        x2 = x + w + 4
        y1 = y - 4
        y2 = y + h + 4
        rect = canvas.create_rectangle(x1, y1, x2, y2, outline='#39FF14') # green
        label = canvas.create_text(x-4, y-15, text=f'{prediction} ({rounded}%)', anchor=W) # prediction text
        rects_and_labels.append(rect)
        rects_and_labels.append(label)

    # writing detected chars in guess frame
    write_chars(guessed_chars)
    # cv.imshow('prediction', blank)

def write_chars(chars):
    # clearing anything in guess frame
    for widget in guess_frame.winfo_children():
        widget.destroy()

    output = ""
    guess_label = Label(guess_frame, text=f'You wrote: {output.join(chars)}')
    guess_label.pack(anchor=NW)


def clear():
    global rects_and_labels
    canvas.delete('all')
    blank.fill(255) # fills array with white pixels
    rects_and_labels = []


# main containers
top_frame = Frame(tk, bg='white', width=FRAME_WIDTH, height=50, relief='ridge', borderwidth=3)
middle_frame = Frame(tk, bg='white', width=FRAME_WIDTH, height=(FRAME_HEIGHT-60)) # , relief='ridge', borderwidth=3
bottom_frame = Frame(tk, bg='white', width=FRAME_WIDTH, height=40) # just for white space at bottom

top_frame.grid(row=0, column=0, sticky='nsew')
top_frame.grid_rowconfigure(0, weight=1)
top_frame.grid_columnconfigure(0, weight=1)
middle_frame.grid(row=1)
bottom_frame.grid(row=2, column=0, sticky='nsew')

# top frame
clear_btn = Button(top_frame, text="CLEAR", command=clear, highlightbackground='yellow')
quit_btn = Button(top_frame, text="QUIT", command=tk.destroy, highlightbackground='red')
clear_btn.pack(side=LEFT, padx=170)
quit_btn.pack(side=RIGHT)

# bottom frame
predict_btn = Button(bottom_frame, text='PREDICT', command=predict_drawing, highlightbackground='#39FF14')
predict_btn.pack(side=LEFT, padx=170)

# middle frame
canvas_frame = Frame(middle_frame, bg='white', width=(FRAME_WIDTH/2), height=(FRAME_HEIGHT-60), relief='ridge', borderwidth=3) # , relief='ridge', borderwidth=3
guess_frame = Frame(middle_frame, bg='white', width=(FRAME_WIDTH/2), height=(FRAME_HEIGHT-60), relief='ridge', borderwidth=3)
guess_frame.pack_propagate(0)
canvas_frame.grid(row=0, column=0)
guess_frame.grid(row=0, column=1)

message = Label(canvas_frame, text='Pass and Drag to Draw', relief=RIDGE) # , font=font.Font(family='Times New Roman')
message.grid(row=0, column=1, columnspan=1)

canvas = Canvas(canvas_frame, width=(FRAME_WIDTH/2 - 12), height=(FRAME_HEIGHT-96), bg='white', cursor='plus')
canvas.bind('<B1-Motion>', paint)
canvas.grid(row=1, columnspan=3)

blank = np.zeros((int(FRAME_HEIGHT-96), int(FRAME_WIDTH/2 - 12), 3), dtype='uint8') # array of black pixels
blank.fill(255) # makes pixels white


tk.mainloop()
# cv.waitKey(0)