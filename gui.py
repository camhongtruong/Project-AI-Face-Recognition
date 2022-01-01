from tkinter import Entry, Label, Tk, Frame, BOTH, font, messagebox, Button
from PIL import ImageTk, Image
import build_dataset as bd
import train_face as tr
import face as recognize

#lớp Gui kế thừa lớp Frame
class Gui(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent, background="#FDFCDC")
        self.parent=parent
        self.initUI()


    def initUI(self):
        self.parent.title("Điểm danh sinh viên")
        self.pack(fill=BOTH, expand=1)

        #Title
        lTitle=Label(self, text="Face Recognition", fg="red", bg="#FDFCDC", font=('Arial',15,'bold'))
        lTitle.grid(row=0, column=1)

        #Logo
        img=ImageTk.PhotoImage(Image.open("images\logo.png"))
        panel=Label(self, image=img,  bg="#FDFCDC")
        panel.image=img
        panel.grid(row=1, column=1)

        #tạo hướng dẫn
        l1=Label(self,text="Step 1. Build dataset", bg="#FDFCDC", font=('Arial',10))
        l1.grid(row=2, column=1)

        l2=Label(self,text="Step 2. Train faces", bg="#FDFCDC", font=('Arial',10))
        l2.grid(row=3, column=1)

        l3=Label(self,text="Step 3. Face recognize", bg="#FDFCDC", font=('Arial',10))
        l3.grid(row=4, column=1)

        #tạo input để nhập họ tên và lớp
        lName=Label(self, text="Name", bg="#FDFCDC", font=('Arial',12))
        lName.grid(row=5, column=0)

        lClass=Label(self, text="ID", bg="#FDFCDC", font=('Arial',12))
        lClass.grid(row=6, column=0)

        eName=Entry(self, width=28)
        eName.grid(row=5, column=1)

        eId=Entry(self, width=28)
        eId.grid(row=6, column=1)

        def run_build_dataset():
            if eName.get()=="" or eId.get()=="":
                messagebox.showinfo("Warning","Please provide Name and ID!")
            else:
                messagebox.showinfo("Instruction","Please press k to capture your image. Program needs at least 5 images. Press q to turn off camera.")
                bd.build_dataset(eName.get(), eId.get())

        def show_train_success():
            tr.train()
            messagebox.showinfo("Result","Training faces is successful!!!")

        builtBtn=Button(self, text="Build dataset", width=15, command=run_build_dataset, bg="#3B559F",fg="white")
        builtBtn.grid(row=7, column=0)
        trainBtn=Button(self, text="Train faces", width=15, command=show_train_success, bg="#4C4141", fg="white")
        trainBtn.grid(row=7, column=1)
        recogBtn=Button(self, text="Face recognize", width=15, command=recognize.recognize, bg="#D24C3E",fg="white")
        recogBtn.grid(row=7, column=2)


root=Tk()
root.geometry("405x250+480+200")
app=Gui(root)
root.mainloop()
