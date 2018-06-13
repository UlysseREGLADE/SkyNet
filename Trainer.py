import tkinter as tk
from tkinter.filedialog import *
import SkyNet.Graphics as gr

import os
import shutil

import time

import tensorflow as tf

class Trainer(tk.Tk):

    def __init__(self, title):

        tk.Tk.__init__(self)
        self.title(title)

        self.creat_output_panel()
        self.creat_control_panel()

        self.creat_graph()
        self.sess = tf.InteractiveSession()

        self.merged = tf.summary.merge_all()
        self.saver = tf.train.Saver()
        self.new_save("Saves/auto_save")

        self.mainloop()

    def creat_graph(self):
        l_variable = tf.Variable([0])
        pass

    def creat_output_panel(self):
        pass

    def creat_control_panel(self):
        l_command_fram = tk.LabelFrame(self, text="Command panel", padx=20, pady=20)
        l_command_fram.pack(side = TOP, padx = 10, pady = 10)
        l_buttons_fram = tk.Frame(l_command_fram)
        l_buttons_fram.pack(side = BOTTOM)
        l_text_fram = tk.Frame(l_command_fram)
        l_text_fram.pack(side = BOTTOM)

        l_save_button = tk.Button(l_buttons_fram, text ='New save', command = self.new_save)
        l_save_button.pack(side = LEFT, padx = 10, pady = 10)
        l_load_button = tk.Button(l_buttons_fram, text ='Load save', command = self.load_save)
        l_load_button.pack(side = LEFT, padx = 10, pady = 10)
        l_start_button = tk.Button(l_buttons_fram, text ='Start training', command = self.start_training)
        l_start_button.pack(side = LEFT, padx = 10, pady = 10)
        l_stop_button = tk.Button(l_buttons_fram, text ='Pause training', command = self.pause_training)
        l_stop_button.pack(side = LEFT, padx = 10, pady = 10)

        self.l_directory_label = tk.Label(l_text_fram, text="Working diretcory: Saves/auto_save", fg="red")
        self.l_directory_label.pack(side = LEFT, padx = 10, pady = 10)
        self.l_bps_label = tk.Label(l_text_fram, text="Batchs per second: 0")
        self.l_bps_label.pack(side = LEFT, padx = 10, pady = 10)

        self.is_training = False

    def start_training(self):
        if(self.is_training == False):
            if(self.trained_model==False):
                if (os.path.exists(self.saving_path)):
                    shutil.rmtree(self.saving_path)
                os.makedirs(self.saving_path)
                self.trained_model==True
                self.l_directory_label.config(text="Working diretcory: "+self.saving_path, fg="green")
            self.writer = tf.summary.FileWriter(self.saving_path, self.sess.graph)
            self.is_training = True
            self.loop_train()

    def pause_training(self):
        if(self.is_training == True):
            self.writer.close()
            self.save()
            self.is_training = False

    def loop_train(self):
        l_starting_time = time.time()

        l_summary = self.train()
        self.training_number += 1
        if(l_summary!=None):
            self.writer.add_summary(l_summary, self.training_number)
        if(self.training_number%100==0):
            self.save()

        l_ending_time = time.time()

        if(self.is_training):
            self.l_bps_label.config(text = "Batchs per second: "+str(int(1/(l_ending_time-l_starting_time))))
            self.after(1, self.loop_train)
        else:
            self.l_bps_label.config(text = "Batchs per second: 0")

    def new_save(self, path=None):
        if(not self.is_training):
            self.trained_model = False
            if(path==None):
                self.saving_path = askdirectory(initialdir="Saves")
            else:
                self.saving_path = path
            self.saving_name = os.path.basename(self.saving_path)
            self.training_number = 0
            self.sess.run(tf.global_variables_initializer())
            self.l_directory_label.config(text="Working diretcory: "+self.saving_path, fg="red")

    def load_save(self):
        if(not self.is_training):
            self.trained_model = True
            self.saving_path = askdirectory(initialdir="Saves")
            l_saving_path = tf.train.latest_checkpoint(self.saving_path)
            self.saving_name = os.path.basename(l_saving_path)
            self.training_number = int(str.split(self.saving_name, "-")[1])
            self.saver.restore(self.sess, l_saving_path)
            self.l_directory_label.config(text="Working diretcory: "+self.saving_path, fg="green")


    def save(self):
        self.saver.save(self.sess, os.path.join(self.saving_path, self.saving_name), self.training_number)

    def batch(self, training=False):
        return None

    def train(self, batch=None):
        return None

    def destroy(self):
        if(self.is_training):
            self.writer.close()
            self.save()
        if(not self.sess._closed):
            self.sess.close()
        tk.Tk.destroy(self)
