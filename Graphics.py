import tkinter as tk
from PIL import Image, ImageTk
import numpy as np

class ImageDisplayer(tk.Canvas):

    #Surcharge du constructeur
    def __init__(self, window, width, heigth, image = None):
        self.width, self.heigth = width, heigth
        if(image == None):
            image = np.zeros((heigth, width, 3), dtype = np.uint8)
        tk.Canvas.__init__(self, window, width = width, height = heigth)
        ImageDisplayer.display(self, image)

    #Prend un talbeau numpy [Hauteur, Largeur, 3] ou [Hauteur, Largeur, 1]
    def display(self, image, float_max=1):
        if(image.dtype != np.dtype(np.uint8)):
            image = np.array(image*255/float_max, dtype=np.uint8)
        if(len(image.shape)<3):
            image = np.reshape([image.shape[0], image.shape[1], 1])
        if(image.shape[2] == 1):
            image = image.squeeze(axis=2)
            image = np.stack((image, image, image), axis=2)
        image = Image.frombytes('RGB', (image.shape[1], image.shape[0]), image.astype('b').tostring())
        self.mImageTk = ImageTk.PhotoImage(image=image)
        self.create_image(0, 0, image=self.mImageTk, anchor=tk.NW)

class DrawingZone(ImageDisplayer):

    def __init__(self, window, width, heigth, image = None):
        ImageDisplayer.__init__(self, window, width, heigth, image)
        if(image!=None):
            self.image = image
        else:
            self.image = np.zeros((self.heigth, self.width, 1))
        self.bind('<B1-Motion>', self.draw)
        self.bind('<B3-Motion>', self.gum)

    def draw(self, event):
        #Si la sourie souris a bougee
        if(event != None):
            x, y = event.x, event.y
            if(x>=0+1 and y>=0+1 and x<self.width-1 and y<self.heigth-1):
                self.image[y-1:y+1, x-1:x+1, 0] = 1
        #Puis on actualise le dessin
        self.display()

    def gum(self, event):
        #Si la sourie souris a bougee
        if(event != None):
            x, y = event.x, event.y
            if(x>=0+1 and y>=0+1 and x<self.width-1 and y<self.heigth-1):
                self.image[y-1:y+1, x-1:x+1, 0] = 0
        #Puis on actualise le dessin
        self.display()

    def display(self, image=None):
        if(image!=None):
            self.image = image
        ImageDisplayer.display(self, self.image)

class LogitsDisplayer(tk.Frame):

    def __init__(self, window, label_names, size=None):
        tk.Frame.__init__(self, window)
        if(size==None):
            self.size = len(label_names)
        else:
            self.size = l_size
        self.label_names = np.array(label_names)
        self.labels = []
        for i in range(self.size):
            self.labels.append(tk.Label(self, text=""))
            self.labels[-1].pack(side = tk.BOTTOM)

    def display(self, logits):
        l_sorted_indexs = np.argsort(logits)
        logits, l_label_names = logits[l_sorted_indexs], self.label_names[l_sorted_indexs]
        for i in range(self.size):
            l_text = l_label_names[i] + ": " + str(int(logits[i]*100))+"%"
            l_font = "Times "+str(int(10+25*logits[i]))
            l_color = "#%02x%02x%02x" % (int(128*(1-logits[i])), int(128*(1-logits[i])), int(128*(1-logits[i])))
            self.labels[i].config(text = l_text, font=l_font, fg=l_color)

class BatchDisplayer(ImageDisplayer):

    def display(self, batch, grads="None", boundries=None):
        #On decoupe le batch si ca depasse
        l_size_x, l_size_y = self.width//batch.shape[2], self.heigth//batch.shape[1]
        if(batch.shape[0]>l_size_x*l_size_y):
            batch = batch[0:l_size_x*l_size_y,:,:,:]
            if(str(grads)!="None"):
                grads = grads[0:l_size_x*l_size_y]
        #On trie les images s'il faut
        if(str(grads)!="None"):
            l_sorted_indexs = np.argsort(-grads)
            batch, grads = batch[l_sorted_indexs,:,:,:], grads[l_sorted_indexs]
            if(boundries==None):
                grads = grads-grads[-1]
            else:
                grads = grads-boundries[0]
            if(grads[0]!=0):
                if(boundries==None):
                    grads = grads/grads[0]
                else:
                    grads = grads/(boundries[1]-boundries[0])
            if(batch.shape[3]!=3):
                batch = batch.squeeze(axis=3)
                batch = np.stack((batch, batch, batch), axis=3)
            for i in range(batch.shape[0]):
                batch[i,:,:,0] = batch[i,:,:,0]*(1-grads[i])
                batch[i,:,:,1] = batch[i,:,:,1]*grads[i]
                batch[i,:,:,2] = batch[i,:,:,2]*0
        #Creation de l'image a partir du batch
        l_image = np.zeros((self.heigth, self.width, batch.shape[3]))
        for i in range(l_size_y):
            for j in range(l_size_x):
                if(l_size_x*i+j < batch.shape[0]):
                    l_image[batch.shape[1]*i:batch.shape[1]*(i+1),batch.shape[2]*j:batch.shape[2]*(j+1),:] = batch[l_size_x*i+j,:,:,:]
        super().display(l_image)

class ConfMatrix(tk.Canvas):

    SQUARE_SIDE = 24
    FONT_SIZE = int(SQUARE_SIDE/2.6)

    def __init__(self, window, size, label_names=None, title=None):
        tk.Canvas.__init__(self, window, width=(size+1)*self.SQUARE_SIDE, height=(size+1)*self.SQUARE_SIDE)
        self.size = size
        if(label_names==None):
            self.label_names = []
            for i in range(self.size):
                self.label_names.append(str(i))
        else:
            self.label_names = label_names
        self.display()

    def display(self, labels="None", logits="None"):
        #Affichage des titre et du fond
        self.delete("all")
        self.create_rectangle(1, 1, (self.size+1)*self.SQUARE_SIDE, (self.size+1)*self.SQUARE_SIDE, fill='black')
        self.create_text(0.5*self.SQUARE_SIDE, self.SQUARE_SIDE*0.5,fill="white",font="Times "+str(self.FONT_SIZE), text="lo\la", anchor=tk.CENTER)
        for i in range(self.size):
            self.create_text((i+1.5)*self.SQUARE_SIDE, self.SQUARE_SIDE*0.5,fill="white",font="Times "+str(self.FONT_SIZE), text=self.label_names[i], anchor=tk.CENTER)
            self.create_text(self.SQUARE_SIDE*0.5, (i+1.5)*self.SQUARE_SIDE,fill="white",font="Times "+str(self.FONT_SIZE), text=self.label_names[i], anchor=tk.CENTER)
        #Creation et affichage de la matrice de confusion
        if(str(labels)!="None" and str(logits)!="None"):
            l_batch_size = labels.shape[0]
            labels = labels.reshape((l_batch_size, 1, self.size))
            labels = np.lib.pad(labels, ((0,0), (0,self.size-1), (0,0)), 'edge')
            logits = logits.reshape((l_batch_size, self.size, 1))
            logits = np.lib.pad(logits, ((0,0), (0,0), (0,self.size-1)), 'edge')
            l_conf_mat = np.sum(labels*logits, axis=0)/np.sum(labels, axis=0)
            for i in range(self.size):
                for j in range(self.size):
                    l_color = "#%02x%02x%02x" % (int(255*(1-l_conf_mat[i,j])), int(255*l_conf_mat[i,j]), 0)
                    l_text = str(int(l_conf_mat[i,j]*100))+"%"
                    self.create_rectangle((i+1)*self.SQUARE_SIDE, (j+1)*self.SQUARE_SIDE, (i+2)*self.SQUARE_SIDE, (j+2)*self.SQUARE_SIDE, fill=l_color)
                    self.create_text((i+1.5)*self.SQUARE_SIDE, (j+1.5)*self.SQUARE_SIDE,fill="black",font="Times "+str(self.FONT_SIZE), text=l_text, anchor=tk.CENTER)


class GraphDisplayer(tk.Canvas):

    #Constante d'affichage
    MARGIN = 0.15

    #Surcharge du constructeur
    def __init__(self, window, width, heigth, title=None, xlabel="", ylabel="", color="red"):
        tk.Canvas.__init__(self, window, width = width, height = heigth)
        self.width, self.heigth = width, heigth
        self.color = color
        GraphDisplayer.display(self, [np.array([0]), np.array([0])], title, xlabel, ylabel)

    #prend un tableau tuple de 2 tableaux numpy
    def display(self, curve, title=None, xlabel="", ylabel=""):
        l_max_x, l_min_x = np.max(curve[0]), np.min(curve[0])
        l_max_y, l_min_y = np.max(curve[1]), np.min(curve[1])
        if(l_max_x==l_min_x):
            l_max_x+=1
        if(l_max_y==l_min_y):
            l_max_y+=1
        l_step_x, l_step_y = (1-2*self.MARGIN)*self.width/(l_max_x-l_min_x), (1-2*self.MARGIN)*self.heigth/(l_max_y-l_min_y)
        l_mar, l_mar_inv = self.MARGIN, 1-self.MARGIN
        #Actualisation du dessin
        self.delete("all")
        self.create_rectangle(l_mar*self.width, l_mar*self.heigth, l_mar_inv*self.width, l_mar_inv*self.heigth, fill='white')
        if(title != None):
            self.create_text(0.5*self.width, self.MARGIN*self.heigth,fill="black",font="Times 8", text=title, anchor=tk.S)
        self.create_text(l_mar*self.width, l_mar_inv*self.heigth,fill="black",font="Times 8", text=str(float(int(l_min_x*100))/100), anchor=tk.NW)
        self.create_text(l_mar_inv*self.width, l_mar_inv*self.heigth,fill="black",font="Times 8", text=str(float(int(l_max_x*100))/100)+xlabel, anchor=tk.NE)
        self.create_text(l_mar*self.width, l_mar_inv*self.heigth,fill="black",font="Times 8", text=str(float(int(l_min_y*100))/100), anchor=tk.SE)
        self.create_text(l_mar*self.width, l_mar*self.heigth,fill="black",font="Times 8", text=str(float(int(l_max_y*100))/100)+ylabel, anchor=tk.NE)

        if(len(curve[0])>=2):
            for i in range(1, len(curve[0])):
                x1 = self.width*l_mar+(curve[0][i-1]-l_min_x)*l_step_x
                y1 = self.heigth*l_mar_inv-(curve[1][i-1]-l_min_y)*l_step_y
                x2 = self.width*l_mar+(curve[0][i]-l_min_x)*l_step_x
                y2 = self.heigth*l_mar_inv-(curve[1][i]-l_min_y)*l_step_y
                self.create_line(x1, y1, x2, y2, width = 1, fill=self.color)

#Permet de monitorer une variable au fil du temps
class Monitor(GraphDisplayer):

    def __init__(self, window, heigth, width, title = "", saturation = 500):
        #Initialisation du moniteur
        self.curve = []
        self.count=0
        self.power = 1
        self.saturation = saturation
        self.title = title
        #Surcharge du constructeur
        super().__init__(window, heigth, width, self.title)

    def display(self, point):
        #Si la liste est trop longue, on raccourcie
        if(len(self.curve)>=self.saturation):
            self.curve = self.curve[::10]
            self.power +=1
        #Si on doit ajouter un element a la liste, on le fait
        self.count+=1
        if(self.count%self.power == 0):
            self.count = 0
            self.curve.append(point)
        #Dans tous les cas, on actualise
        l_curve_y = np.array(self.curve)
        l_curve_x = np.arange(0, len(self.curve), 1)
        l_curve = np.array([l_curve_x, l_curve_y])
        super().display(l_curve, self.title, xlabel="x10^"+str(self.power-1))
