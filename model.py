import keras
from keras import Model

import numpy as np




## This is an encoder AI
class AutoEncoder():
    def __init__(self , NUM_LAYERS , image_size , conv_filter , conv_kernel_size , conv_strides , conv_t_filter , conv_t_kernel_size , conv_t_strides , z_dim):
        ## Constants
        self.num_layers = NUM_LAYERS
        self.image_dim = image_size
        self.conv_filter = conv_filter
        self.conv_kernel_size = conv_kernel_size
        self.conv_stride = conv_strides
        self.conv_t_filter = conv_t_filter
        self.conv_t_kernel_size = conv_t_kernel_size 
        self.conv_t_stride = conv_t_strides
        self.z_dim = z_dim

        ## Building the encoder
        encoder_input = keras.layers.Input(image_size)
        x = encoder_input
        for i in range(self.num_layers):
            conv_layer = keras.layers.Conv2D(
                filters=self.conv_filter[i],
                kernel_size=self.conv_kernel_size[i],
                strides=self.conv_stride[i],
                padding = "same"
            )

            x = conv_layer(x)
            x = keras.layers.LeakyReLU()(x)
        self.shape_before_flattening = keras.backend.int_shape(x)[1:]
        x = keras.layers.Flatten()(x)
        encoder_output = keras.layers.Dense(self.z_dim)(x)
        self.encoder = Model(encoder_input , encoder_output)

        ## Building the Decoder
        decoder_input = keras.layers.Input((self.z_dim,))
        x = keras.layers.Dense(np.prod(self.shape_before_flattening))(decoder_input)
        x = keras.layers.Reshape(self.shape_before_flattening)(x)
        for i in range(self.num_layers):
            conv_t_layer = keras.layers.Conv2DTranspose(
                filters=self.conv_t_filter[i],
                kernel_size=self.conv_t_kernel_size[i],
                strides=self.conv_t_stride[i],
                padding="same"
            )
            x = conv_t_layer(x)
            if i<self.num_layers-1:
                x = keras.layers.LeakyReLU()(x)
            else:
                x=keras.layers.Activation('sigmoid')(x)
        decode_output = x
        self.decoder = Model(decoder_input , decode_output)

        model_input = encoder_input
        model_output = self.decoder(encoder_output)
        self.model = Model(model_input , model_output)
    
    def train(self, X):
        optimizer = keras.optimizers.Adam(5e-4)
        self.model.compile(optimizer=optimizer , loss=self.r_loss)
        self.model.fit(X , X , 128 , 20 , shuffle=True)

    @keras.saving.register_keras_serializable()
    def r_loss(self, y_true , y_pred):
        return keras.backend.mean(keras.backend.square(y_true-y_pred) , axis=[1,2,3]) 

class Vanilla_GAN():
    def __init__(self, input_dim, d_filters, d_kernel, d_strides, d_batch_norm_momentum, d_activation, d_dropout_rate, d_lr,
                 g_init_dense, g_upsample, g_filters, g_kernel, g_stride, g_batch_norm_momentum, g_activation, g_dropout_rate, g_lr,
                 optimizer, z_dim):
        self.input_dim = input_dim
        self.d_filters = d_filters
        self.d_kernel = d_kernel
        self.d_strides = d_strides
        self.d_batch_norm_momentum = d_batch_norm_momentum
        self.d_activation = d_activation
        self.d_dropout_rate = d_dropout_rate
        self.d_lr = d_lr
        self.g_init_dense = g_init_dense
        self.g_upsample = g_upsample
        self.g_filters = g_filters
        self.g_kernel = g_kernel
        self.g_stride = g_stride
        self.g_batch_norm_momentume = g_batch_norm_momentum
        self.g_activation = g_activation
        self.g_dropout_rate = g_dropout_rate
        self.g_lr = g_lr
        self.optimizer = optimizer
        self.z_dim = z_dim

        self.build_gan()
        
    def build_gan(self):
        self.build_generator()
        self.build_discriminator()
        self.discriminator.compile(optimizer=keras.optimizers.RMSprop(0.0008), loss = 'binary_crossentropy', metrics=['accuracy'])
        self.discriminator.trainable = False
        model_input = keras.layers.Input(shape=(self.z_dim,), name = 'model_input')
        model_output = self.discriminator(self.generator(model_input))
        self.model = keras.Model(model_input , model_output)
        self.model.compile(
            optimizer=keras.optimizers.RMSprop(0.0004), loss='binary_crossentropy', metrics=['accuracy']
        )

    def build_discriminator(self):
        discriminator_input = keras.layers.Input(shape= self.input_dim, name='discriminator_input')
        x=discriminator_input
        for i in range(len(self.d_filters)):
            x = keras.layers.Conv2D(
                filters = self.d_filters[i],
                kernel_size = self.d_kernel[i],
                strides = self.d_strides[i],
                padding = 'same',
                name = 'd_conv_'+str(i)
            )(x)
            if self.d_batch_norm_momentum and i>0:
                x = keras.layers.BatchNormalization(momentum = self.d_batch_norm_momentum)(x)
            x = keras.layers.Activation(self.d_activation)(x)
            if self.d_dropout_rate:
                x = keras.layers.Dropout(rate=self.d_dropout_rate)(x)
        x = keras.layers.Flatten()(x)
        discriminator_output = keras.layers.Dense(1, activation='sigmoid')(x)
        self.discriminator = keras.Model(discriminator_input , discriminator_output)
    
    def build_generator(self):
        generator_input = keras.layers.Input(shape=(self.z_dim,), name='generator_input')
        x = generator_input
        x = keras.layers.Dense(np.prod(self.g_init_dense))(x)
        if self.g_batch_norm_momentume:
            x = keras.layers.BatchNormalization(momentum=self.g_batch_norm_momentume)(x)
        x = keras.layers.Activation(self.g_activation)(x)
        x = keras.layers.Reshape(self.g_init_dense)(x)
        if self.g_dropout_rate:
            x = keras.layers.Dropout(rate=self.g_dropout_rate)(x)

        for i in range(len(self.g_filters)):
            # x = keras.layers.UpSampling2D()(x)
            x = keras.layers.Conv2DTranspose(
                filters=self.g_filters[i],
                kernel_size= self.g_kernel[i],
                padding='same',
                name = 'g_conv_'+str(i),
                strides=self.g_stride[i]
            )(x)
            if i<len(self.g_filters)-1:
                if self.g_batch_norm_momentume:
                    x = keras.layers.BatchNormalization(momentum=self.g_batch_norm_momentume)(x)
                x = keras.layers.Activation('relu')(x)
            else:
                x = keras.layers.Activation('tanh')(x)

        generator_output = x
        self.generator = keras.Model(generator_input , generator_output)
    
    def train_discriminator(self, x_train, batch_size):
        valid = np.ones((batch_size,1))
        fake = np.zeros((batch_size,1))
        idx = np.random.randint(0,x_train.shape[0], batch_size)
        true_images = x_train[idx]
        self.discriminator.train_on_batch(true_images , valid)

        noise = np.random.normal(0,1,(batch_size,self.z_dim))
        gen_images = self.generator.predict(noise)
        self.discriminator.train_on_batch(gen_images, fake)

    def train_generator(self, batch_size):
        valid = np.ones((batch_size,1))
        noise = np.random.normal(0,1,(batch_size,self.z_dim))
        self.model.train_on_batch(noise,valid)

    def train(self, x_train):
        epochs = 2000
        batch_size = 64
        for epoch in range(epochs):
            print(f"Epoch : {epoch}")
            self.train_discriminator(x_train, batch_size)
            self.train_generator(batch_size)
        pass

class WGAN_WC():
    def __init__(self, input_dim, d_filters, d_kernel, d_strides, d_batch_norm_momentum, d_activation, d_dropout_rate, d_lr,
                 g_init_dense, g_upsample, g_filters, g_kernel, g_stride, g_batch_norm_momentum, g_activation, g_dropout_rate, g_lr,
                 optimizer, z_dim):
        self.input_dim = input_dim
        self.d_filters = d_filters
        self.d_kernel = d_kernel
        self.d_strides = d_strides
        self.d_batch_norm_momentum = d_batch_norm_momentum
        self.d_activation = d_activation
        self.d_dropout_rate = d_dropout_rate
        self.d_lr = d_lr
        self.g_init_dense = g_init_dense
        self.g_upsample = g_upsample
        self.g_filters = g_filters
        self.g_kernel = g_kernel
        self.g_stride = g_stride
        self.g_batch_norm_momentume = g_batch_norm_momentum
        self.g_activation = g_activation
        self.g_dropout_rate = g_dropout_rate
        self.g_lr = g_lr
        self.optimizer = optimizer
        self.z_dim = z_dim

        self.build_gan()
        
    def build_gan(self):
        self.build_generator()
        self.build_discriminator()
        self.discriminator.compile(
            optimizer=keras.optimizers.RMSprop(0.00005), loss=self.wasserstein
        )
        self.discriminator.trainable = False
        model_input = keras.layers.Input(shape=(self.z_dim,), name = 'model_input')
        model_output = self.discriminator(self.generator(model_input))
        self.model = keras.Model(model_input , model_output)
        self.model.compile(
            optimizer=keras.optimizers.RMSprop(0.00005), loss=self.wasserstein
        )

    def build_discriminator(self):
        discriminator_input = keras.layers.Input(shape= self.input_dim, name='discriminator_input')
        x=discriminator_input
        for i in range(len(self.d_filters)):
            x = keras.layers.Conv2D(
                filters = self.d_filters[i],
                kernel_size = self.d_kernel[i],
                strides = self.d_strides[i],
                padding = 'same',
                name = 'd_conv_'+str(i)
            )(x)
            if self.d_batch_norm_momentum and i>0:
                x = keras.layers.BatchNormalization(momentum = self.d_batch_norm_momentum)(x)
            x = keras.layers.Activation(self.d_activation)(x)
            if self.d_dropout_rate:
                x = keras.layers.Dropout(rate=self.d_dropout_rate)(x)
        x = keras.layers.Flatten()(x)
        discriminator_output = keras.layers.Dense(1)(x)
        self.discriminator = keras.Model(discriminator_input , discriminator_output)
    
    def build_generator(self):
        generator_input = keras.layers.Input(shape=(self.z_dim,), name='generator_input')
        x = generator_input
        x = keras.layers.Dense(np.prod(self.g_init_dense))(x)
        if self.g_batch_norm_momentume:
            x = keras.layers.BatchNormalization(momentum=self.g_batch_norm_momentume)(x)
        x = keras.layers.Activation(self.g_activation)(x)
        x = keras.layers.Reshape(self.g_init_dense)(x)
        if self.g_dropout_rate:
            x = keras.layers.Dropout(rate=self.g_dropout_rate)(x)

        for i in range(len(self.g_filters)):
            # x = keras.layers.UpSampling2D()(x)
            x = keras.layers.Conv2DTranspose(
                filters=self.g_filters[i],
                kernel_size= self.g_kernel[i],
                padding='same',
                name = 'g_conv_'+str(i),
                strides=self.g_stride[i]
            )(x)
            if i<len(self.g_filters)-1:
                if self.g_batch_norm_momentume:
                    x = keras.layers.BatchNormalization(momentum=self.g_batch_norm_momentume)(x)
                x = keras.layers.Activation('relu')(x)
            else:
                x = keras.layers.Activation('tanh')(x)

        generator_output = x
        self.generator = keras.Model(generator_input , generator_output)
    
    def train_discriminator(self, x_train, batch_size):
        valid = np.ones((batch_size,1))
        fake = -np.ones((batch_size,1))
        idx = np.random.randint(0,x_train.shape[0], batch_size)
        true_images = x_train[idx]
        self.discriminator.train_on_batch(true_images , valid)

        noise = np.random.normal(0,1,(batch_size,self.z_dim))
        gen_images = self.generator.predict(noise)
        self.discriminator.train_on_batch(gen_images, fake)
        for l in self.discriminator.layers:
            weights = l.get_weights()
            weights = [np.clip(w, -0.01, 0.01) for w in weights]
            l.set_weights(weights)

    def train_generator(self, batch_size):
        valid = np.ones((batch_size,1))
        noise = np.random.normal(0,1,(batch_size,self.z_dim))
        self.model.train_on_batch(noise,valid)

    def train(self, x_train):
        epochs = 2000
        batch_size = 64
        for epoch in range(epochs):
            print(f"Epoch : {epoch}")
            for i in range(5):
                self.train_discriminator(x_train, batch_size)
            self.train_generator(batch_size)
        pass

    def wasserstein(self, y_true, y_pred):
        return -keras.backend.mean(y_true * y_pred)
