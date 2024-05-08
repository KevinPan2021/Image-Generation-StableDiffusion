application_name = 'Image Generation'
# pyqt packages
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QFileDialog, QLabel
from PyQt5.QtCore import pyqtSignal, QThread

from skimage.transform import resize
import sys
import numpy as np
import torch
from PIL import Image
from transformers import CLIPTokenizer
import re
import os


from qt_main import Ui_Application
from main import compute_device
from ddpm import DDPMSampler
from pipeline import preload_models, rescale, get_time_embedding

width = 512
height = 512
latents_width = width//8
latents_height = height//8


class Generator(QThread):
    parent_class = None
    iterationUpdated = pyqtSignal(int)
    work_complete_signal = pyqtSignal()
    
    def __init__(self, prompt, uncond_prompt, input_image = None, strength=0.8, do_cfg=True, cfg_scale=7.5, sampler_name ='ddpm', n_inference=50, 
                models={}, seed=None, device=None, idle_device=None, tokenizer=None):
        super().__init__()
        self.prompt = prompt
        self.uncond_prompt = uncond_prompt
        self.input_image = input_image
        
        self.strength = strength
        self.do_cfg = do_cfg
        self.cfg_scale = cfg_scale
        self.sampler_name = sampler_name
        self.n_inference = n_inference
        self.models = models
        self.seed = seed
        self.device = device
        self.idle_device = idle_device
        self.tokenizer = tokenizer
        
        
    @torch.no_grad
    def run(self):
        if not (0 < self.strength <= 1):
            raise ValueError('stength must be between [0,1]')
        
        if self.idle_device:
            to_idle = lambda x:x.to(self.idle_device)
        else:
            to_idle = lambda x:x
        
        
        generator = torch.Generator(device=self.device)
        if self.seed is None:
            generator.seed()
        else:
            generator.manual_seed(self.seed)
        
        
        clip = self.models['clip']
        clip = clip.to(self.device)
        
        # classifier free guidence
        if self.do_cfg:
            # convert prompt into tokens
            cond_tokens = self.tokenizer.batch_encode_plus([self.prompt], padding='max_length', max_length=77).input_ids
            # (batch, seq_len)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=self.device)
            # (batch, seq_len) -> (batch, seq_len, dim)
            cond_context = clip(cond_tokens)
            
            # convert prompt into tokens
            uncond_tokens = self.tokenizer.batch_encode_plus([self.uncond_prompt], padding='max_length', max_length=77).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=self.device)
            # (batch, seq_len) -> (batch, seq_len, dim)
            uncond_context = clip(uncond_tokens)
            
            # concatenate prompt
            # (2, seq_len, dim)
            context = torch.cat([cond_context, uncond_context])
            
        else:
            # convert prompt into tokens
            cond_tokens = self.tokenizer.batch_encode_plus([self.prompt], padding='max_length', max_length=77).input_ids
            # (batch, seq_len)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=self.device)
            # (1, seq_len, dim)
            context = clip(cond_tokens)
        
        to_idle(clip)
        
        if self.sampler_name == 'ddpm':
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(self.n_inference)
            
        else:
            raise ValueError(f'Unknown Sampler {self.sampler_name}')
        
        latents_shape = (1, 4, latents_height, latents_width)
        
        # if input_image is provider, do Image-to-Image task
        if self.input_image:
            encoder = self.models['encoder']
            encoder.to(self.device)
            
            input_image_tensor = resize(self.input_image, (width, height))
            # (height, width, channel)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=self.device)
            
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            # (height, width, channel) -> (batch, height, width, channel)
            input_image_tensor = input_image_tensor.unsqueeze(0) 
            # (batch, height, width, channel) -> (batch, channel, height, width)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)
            
            # sample noise
            encoder_noise = torch.randn(latents_shape, generator=generator, device=self.device)
            
            # run image through the VAE decoder
            latents = encoder(input_image_tensor, encoder_noise)
            
            sampler.set_strength(strength=self.strength)
            
            latents = sampler.add_noise(latents, sampler.timesteps[0])
            
            to_idle(encoder)
            
        # text-to-image task, start with random noise N(0, I)
        else:
            latents = torch.randn(latents_shape, generator=generator, device=self.device)
        
        # load the diffusion model
        diffusion = self.models['diffusion']
        diffusion.to(self.device)
        
        # denoising
        for i, timestep in enumerate(sampler.timesteps):
            # (1, 320)
            time_embedding = get_time_embedding(timestep).to(self.device)
            
            # (batch, 4, latent_height, latent_width)
            model_input = latents
               
            if self.do_cfg:
                # (batch, 4, latent_height, latent_width) -> (2, batch, 4, latent_height, latent_width)
                model_input = model_input.repeat(2, 1, 1, 1)
                
            # predicted noise by UNet
            model_output = diffusion(model_input, context, time_embedding)
            
            if self.do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = self.cfg_scale * (output_cond - output_uncond) + output_uncond
            
            # remove the predicted noise
            latents = sampler.remove_noise(timestep, latents, model_output)
            
            # emit progress
            self.iterationUpdated.emit(i)
            
        to_idle(diffusion)
        
        # load the decoder
        decoder = self.models['decoder']
        decoder.to(self.device)
        
        images = decoder(latents)
        to_idle(decoder)
        
        images = rescale(images, (-1, 1), (0,255), clamp=True)
        # (batch, channel, height, width) -> (batch, height, width, channel)
        images = images.permute(0, 2, 3, 1)
        images = images.to('cpu', torch.uint8).numpy()
        
        self.parent_class.output_image = images[0]
        self.work_complete_signal.emit()
                
        

# Regular expression to match numbers (including integers and floats)
def is_numeric(input_str):
    numeric_pattern = r'^[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$'
    return bool(re.match(numeric_pattern, input_str))


def show_message(parent, title, message, icon=QMessageBox.Warning):
        msg_box = QMessageBox(icon=icon, text=message)
        msg_box.setWindowIcon(parent.windowIcon())
        msg_box.setWindowTitle(title)
        msg_box.setStyleSheet(parent.styleSheet() + 'color:white} QPushButton{min-width: 80px; min-height: 20px; color:white; \
                              background-color: rgb(91, 99, 120); border: 2px solid black; border-radius: 6px;}')
        msg_box.exec()



# pad the image to square
def pad_to_square(image):
    # Get the height and width of the input image
    height, width = image.shape[:2]

    # Calculate the size of the square
    square_len = max(height, width)

    # Create a new square image filled with zeros
    padded_image = np.zeros((square_len, square_len, 3), dtype=np.uint8)

    # Calculate the starting position to append the original image
    start_h = (square_len - height) // 2
    start_w = (square_len - width) // 2

    # Append the original image to the center of the square image
    padded_image[start_h:start_h+height, start_w:start_w+width] = image

    return padded_image

    
        
        
class QT_Action(Ui_Application, QMainWindow):
    label_image_ClickSig = pyqtSignal()
    
    def __init__(self):
        # system variable
        super(QT_Action, self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)
        self.setWindowTitle(application_name) # set the title
        
        # runtime variable
        self.image_size = 512
        self.input_image = None
        self.model = None
        self.steps = 50
        self.cfg = 8
        self.seed = None
        self.strength = 0.9
        self.output_image = None
        self.pretrained_path = '../pretrained_models/Stable Diffusion/'
        self.tokenizer = CLIPTokenizer(
            self.pretrained_path+'tokenizer_vocab.json', 
            merges_file=self.pretrained_path+'tokenizer_merges.txt'
        )
        
        # load the model
        self.load_model_action()
        
        
        
    # single click to import image
    def mousePressEvent(self, event):
        widget = self.childAt(event.pos())
        
        if widget is None:
            return 
        
        if isinstance(widget, QLabel) and widget.objectName == self.label_input_image.objectName:
            self.label_image_ClickSig.emit()
            
        
    # linking all button/textbox with actions    
    def link_commands(self,):
        self.label_image_ClickSig.connect(self.import_image_action)
        self.lineEdit_steps.editingFinished.connect(self.load_step_action)
        self.lineEdit_seed.editingFinished.connect(self.load_seed_action)
        self.lineEdit_cfg.editingFinished.connect(self.load_cfg_action)
        self.lineEdit_strength.editingFinished.connect(self.load_strength_action)
        self.comboBox_model.activated.connect(self.load_model_action)
        
        self.toolButton_process.clicked.connect(self.process_action)
        self.toolButton_export.clicked.connect(self.export_action)
        
    
    # clicking the import button action
    def import_image_action(self,):
        # show an "Open" dialog box and return the path to the selected file
        filename, _ = QFileDialog.getOpenFileName(None, "Select file", options=QFileDialog.Options())
        
        # didn't select any files
        if filename is None or filename == '': 
            return
    
        # selected .oct or .octa files
        if filename.endswith('.jpg'):
            self.image = Image.open(filename)
            
            self.image = np.array(self.image).astype(np.uint8)
            self.image = pad_to_square(self.image)
            data = self.image
            height, width, channels = data.shape
            q_image = QImage(data.tobytes(), width, height, width*channels, QImage.Format_RGB888)  # Create QImage
            
            qpixmap = QPixmap.fromImage(q_image)  # Convert QImage to QPixmap
            self.label_input_image.setPixmap(qpixmap)
        
        # selected the wrong file format
        else:
            show_message(self, title='Load Error', message='Available file format: .jpg')
            self.import_image_action()
        
        
    
    
    def load_step_action(self):
        steps = self.lineEdit_steps.text()
        
        if is_numeric(steps):
            self.steps = int(steps)
        else:
            self.lineEdit_steps.setText('50')
            self.steps = 50
    
    
    
    def load_seed_action(self):
        seed = self.lineEdit_seed.text()
        
        if is_numeric(seed):
            self.seed = int(seed)
        else:
            self.seed = None
            self.lineEdit_seed.setText('')
        
        
        
    def load_cfg_action(self):
        cfg = self.lineEdit_cfg.text()
        
        if is_numeric(cfg):
            self.cfg = float(cfg)
        else:
            self.lineEdit_cfg.setText('8')
            self.cfg = 8
        
        
    
    def load_strength_action(self):
        strength = self.lineEdit_strength.text()
        
        if is_numeric(strength) and float(strength) < 1.0:
            self.strength = float(strength)
        else:
            self.lineEdit_strength.setText('0.9')
            self.strength = 0.9
        
        
    
    # choosing between models
    def load_model_action(self,):
        self.model_name = self.comboBox_model.currentText()
        
        # load the model
        if self.model_name == 'Stable Diffusion':
            # load the trained model
            self.model = preload_models(self.pretrained_path+'v1-5-pruned-emaonly.ckpt')
            
    
    def update_progress(self, progress):
        progress_percent = int((progress+1)/self.steps*100)
        self.progressBar.setValue(progress_percent)
        
    
    def process_action(self):
        if self.textEdit_prompt.toPlainText() == '':
            show_message(self, title='Process Error', message='Please enter a prompt')
            return
        
        # tokenize prompt and uncond prompt
        prompt = self.textEdit_prompt.toPlainText()
        uncond_prompt = self.textEdit_uncond_prompt.toPlainText()
        
        # generate
        self.generator = Generator(
            prompt,
            uncond_prompt,
            input_image = self.input_image,
            strength = self.strength,
            do_cfg = True,
            cfg_scale = self.cfg,
            sampler_name = 'ddpm',
            n_inference = self.steps,
            seed = self.seed,
            models = self.model,
            device = compute_device(),
            idle_device = 'cpu',
            tokenizer = self.tokenizer
        )
        self.generator.parent_class = self
        self.generator.iterationUpdated.connect(self.update_progress)
        self.generator.work_complete_signal.connect(self.display_output_image)
        
        # generate output image
        self.generator.start()
        
    
    def display_output_image(self):
        height, width, channel = self.output_image.shape
        qimage = QImage(self.output_image.data.tobytes(), width, height, 3*width, QImage.Format_RGB888)
        qpixmap = QPixmap.fromImage(qimage)

        # Display the result in label_detection
        self.label_generated_image.setPixmap(qpixmap)
    
    
    def export_action(self):
        if self.output_image is None:
            show_message(self, title='Export Error', message='Please process first')
            return
        
        # Open a file dialog to select the export folder
        export_folder = QFileDialog.getExistingDirectory(self, "Select Export Folder", os.path.expanduser("~"))
    
        if export_folder:
            image = self.output_image
            
            # Create a PIL Image from the output numpy array
            image = Image.fromarray(image)
    
            # Save the image as .jpg file in the selected folder
            file_path = os.path.join(export_folder, "output_image.jpg")
            image.save(file_path, "JPEG")
    
    
    
            
def main():
    app = QApplication(sys.argv)
    action = QT_Action()
    action.link_commands()
    action.show()
    sys.exit(app.exec_())
    
    
if __name__ == '__main__':
    main()