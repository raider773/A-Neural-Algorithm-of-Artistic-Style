import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np


class neural_transfer_model(tf.keras.Model):       
    def __init__(self,pre_trained_model,rows,cols,content_layers,style_layers,content_weight,style_weight):
        super().__init__()     
        
        self.rows = rows
        self.cols = cols
        
        self.pre_trained_model = pre_trained_model
        self.white_image = tf.Variable(tf.ones(shape = (1,rows,cols,3)))       
        
        self.content_layer = self.pre_trained_model.get_layer(content_layers).output
        self.style_layer = [self.pre_trained_model.get_layer(layer).output for layer in style_layers]        
        
        self.content_model = Model(inputs = self.pre_trained_model.inputs, outputs = self.content_layer)
        self.style_model = Model(inputs = self.pre_trained_model.inputs, outputs = self.style_layer)
        
        self.content_weight = content_weight
        self.style_weight = style_weight
       
    
    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer   
        
    def gram_matrix(self,x):  #Detailed in notebook style_Loss
        features = K.batch_flatten(K.permute_dimensions(x[0],(2,0,1)))
        gram = K.dot(features,K.transpose(features))
        return gram

    def style_loss(self,style,white_image):    #Detailed in notebook style_Loss
        style_gram_matrix = self.gram_matrix(style)
        white_image_gram_matrix = self.gram_matrix(white_image)
        size = self.rows * self.cols
        return tf.reduce_sum(tf.square(style_gram_matrix - white_image_gram_matrix)) / (36 * (size**2))   
    
    def content_loss(self,image):    #Detailed in notebook content_Loss
        base_pred = self.content_model(image)     
        white_pred = self.content_model(self.white_image)         
        return tf.reduce_sum(tf.square(white_pred - base_pred))
    
        
    def train_step(self, data):                
        
        content_data = data[0][0]
        style_data = data[0][1]
        
        with tf.GradientTape() as tape:      
            
            s_loss = tf.zeros(shape=()) #for summing each layer loss     
            
            style_fowards = self.style_model(style_data)
            style_white_noise_fowards = self.style_model(self.white_image)  
            
            for layer_index in range(len(style_fowards)):    
                current_layer_loss = self.style_loss(style_fowards[layer_index],style_white_noise_fowards[layer_index]) 
                #1/amount of layers * current layer loss to give same importance to each layer loss    
                s_loss += (1 / len(style_fowards)) * current_layer_loss  
                        
                
            c_loss = self.content_loss(content_data)
            
            total_loss = self.content_weight * c_loss + self.style_weight * s_loss 
                
        grads = tape.gradient(total_loss, [self.white_image])[0]            
        self.optimizer.apply_gradients([(grads, self.white_image)])  
        return {"total_loss": total_loss,"style_loss": s_loss,"content_loss": c_loss}