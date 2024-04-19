"""
Scribble Architect allows to transforms simple doodles into architectural works!
Author: Samuel Dubois
Any remark/suggestion: sdu@bbri.be
"""

from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from PyQt6 import uic

import widgets as wid
import resources as res
from lcm import *
from PIL import Image, ImageOps

import torch
import os
import gc

# Params
IMG_W = 700
IMG_H = 500


def new_dir(dir_path):
    """
    Simple function to verify if a directory exists and if not creating it
    :param dir_path: (str) the path to check
    :return:
    """
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


def scene_to_image(viewer):
    # Define the size of the image (same as the scene's bounding rect)
    image = QImage(viewer.viewport().size(), QImage.Format.Format_ARGB32_Premultiplied)

    # Create a QPainter to render the scene into the QImage
    painter = QPainter(image)
    viewer.render(painter)
    painter.end()

    file_path = 'input.png'
    image.save(file_path)

    # Open the image and convert it to RGB mode
    pil_img = Image.open(file_path).convert('RGB')

    # Invert the colors
    inverted_img = ImageOps.invert(pil_img)

    # Save the inverted image
    inverted_file_path = 'inverted_input.png'
    inverted_img.save(inverted_file_path)

    return inverted_img


class PaintLCM(QMainWindow):
    def __init__(self, is_dark_theme):
        super().__init__()

        basepath = os.path.dirname(__file__)
        basename = 'interface'
        uifile = os.path.join(basepath, '%s.ui' % basename)
        uic.loadUi(uifile, self)

        self.setWindowTitle("ScribbleArchitect!")

        # add actions to action group
        ag = QActionGroup(self)
        ag.setExclusive(True)
        ag.addAction(self.brush_action)
        ag.addAction(self.eraser_action)

        self.img_dim = (IMG_W, IMG_H)
        self.canvas = wid.Canvas(self.img_dim)
        self.horizontalLayout_4.addWidget(self.canvas)

        self.result_canvas = wid.simpleCanvas(self.img_dim)
        self.horizontalLayout_4.addWidget(self.result_canvas)

        # loads models
        self.models = model_list
        self.models_ids = model_ids

        # initial parameters
        self.infer = load_models()
        self.im = None
        self.original_parent = None

        # pre-img parameters
        self.simple_prompts = ['A building architectural render',
                               'A building artistic architectural drawing',
                               'A city',
                               'The cross section of a building',
                               'A facade elevation',
                               'A floor plan',
                               'The drawing of an interior',
                               'Interior architectural render',
                               'Isometric building',
                               'Cutaway of a building showing its interior, isometric',
                               'Ground plan landscape architect'
                               ]
        self.example_paths = [res.find('img/examples/building_default.png'),
                              res.find('img/examples/building_default.png'),
                              res.find('img/examples/city_default.png'),
                              res.find('img/examples/cross_default.png'),
                              res.find('img/examples/facade_default.png'),
                              res.find('img/examples/floorplan_default.png'),
                              res.find('img/examples/interior_default.png'),
                              res.find('img/examples/interior_default.png'),
                              res.find('img/examples/iso_default.png'),
                              res.find('img/examples/cutaway_default.png'),
                              res.find('img/examples/facade_default.png')
                              ]


        # base_dir = res.find('img/AI_ref_images')
        base_dir = res.find('img/AI_ref_images_bo')
        self.type_names = []  # To store the type names as read from folder names
        self.all_ip_styles = []  # To store style names
        self.all_ip_prompts = []  # To store corresponding prompts
        self.all_ip_paths = []  # To store image paths

        # Iterate through each type subfolder in the base directory
        for idx, type_folder in enumerate(sorted(os.listdir(base_dir))):
            type_path = os.path.join(base_dir, type_folder)
            if os.path.isdir(type_path):
                self.type_names.append(type_folder)

                styles = []
                prompts = []
                image_paths = []

                # Iterate through each file in the subfolder
                for file in sorted(os.listdir(type_path)):
                    if file.endswith('.png'):
                        image_path = os.path.join(type_path, file)
                        image_paths.append(image_path)

                        # Attempt to find a matching text file for prompts
                        prompt_path = image_path.replace('.png', '.txt')
                        prompt = ''
                        if os.path.exists(prompt_path):
                            with open(prompt_path, 'r') as f:
                                prompt = f.read().strip()
                        prompts.append(prompt)

                        # Assume style name is the file name without extension
                        style_name = os.path.splitext(file)[0]
                        styles.append(style_name)

                self.all_ip_styles.append(styles)
                self.all_ip_prompts.append(prompts)
                self.all_ip_paths.append(image_paths)

        self.comboBox.addItems([f'Type {i + 1} ({name})' for i, name in enumerate(self.type_names)])

        # After initializing the comboBox and adding items with icons, set the icon size
        desired_icon_width = 80  # Adjust the width as needed
        desired_icon_height = 80  # Adjust the height as needed

        self.comboBox_style.setIconSize(QSize(desired_icon_width, desired_icon_height))

        # Loop through the flattened lists and add items with icons to the comboBox_style
        for style, img_path in zip(self.all_ip_styles[0], self.all_ip_paths[0]):
            icon = QIcon(img_path)
            self.comboBox_style.addItem(icon, style)

        # specific variables
        if not path.exists(cache_path):
            os.makedirs(cache_path, exist_ok=True)

        # seed value
        # Assuming a maximum value of 9999 for demonstration; adjust as needed
        validator = QIntValidator(0, 9999, self)

        # Apply the validator to the line edit
        self.lineEdit_seed.setValidator(validator)

        # connections
        self.brush_action.triggered.connect(lambda: self.canvas.set_tool('brush'))
        self.eraser_action.triggered.connect(lambda: self.canvas.set_tool('eraser'))
        self.color_action.triggered.connect(self.reset_canvas)
        self.export_action.triggered.connect(self.save_output)

        # self.actionLoad_IP_Adapter_reference_image.triggered.connect(self.define_ip_ref)
        self.pushButton.clicked.connect(self.update_image)
        self.pushButton_example.clicked.connect(self.import_example)

        # self.checkBox_ip.stateChanged.connect(self.toggle_ip)

        # when editing canvas --> update inference
        self.canvas.endDrawing.connect(self.update_brush_stroke)

        # combobox
        self.comboBox.currentIndexChanged.connect(self.change_predefined_prompts)
        self.comboBox_style.currentIndexChanged.connect(self.change_style)

        # Connect the sliders to the update_image function
        self.step_slider.valueChanged.connect(self.update_image)
        self.cfg_slider.valueChanged.connect(self.update_image)
        self.strength_slider.valueChanged.connect(self.update_image)

        # Connect the text edit to the update_image function
        self.textEdit.setWordWrapMode(QTextOption.WrapMode.WordWrap)
        self.textEdit.setText(self.all_ip_prompts[0][0])

        self.textEdit_negative.setWordWrapMode(QTextOption.WrapMode.WordWrap)

        # checkboxes
        self.checkBox_sp.stateChanged.connect(self.change_style)

        # other actions
        self.actionAdvanced_options.triggered.connect(self.toggle_dock_visibility)

        # seed edit
        self.lineEdit_seed.textChanged.connect(self.update_image)

        # drawing ends

        if is_dark_theme:
            suf = '_white_tint'
            suf2 = '_white'
        else:
            suf = ''

        self.add_icon(res.find(f'img/brush{suf}.png'), self.brush_action)
        self.add_icon(res.find(f'img/eraser{suf}.png'), self.eraser_action)
        self.add_icon(res.find(f'img/mop{suf}.png'), self.color_action)
        self.add_icon(res.find(f'img/save_as{suf}.png'), self.export_action)

        # run first inference
        self.update_image()

    # general functions __________________________________________
    def toggle_dock_visibility(self):
        if self.dockWidget_2.isVisible():
            self.dockWidget_2.hide()
        else:
            self.dockWidget_2.show()

    def reset_canvas(self):
        self.canvas.clean_scene()

    def add_icon(self, img_source, pushButton_object):
        """
        Function to add an icon to a pushButton
        """
        pushButton_object.setIcon(QIcon(img_source))

    def save_output(self):
        # add code for file dialog
        file_path, _ = QFileDialog.getSaveFileName(
            None, "Save Image", "", "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg *.JPEG)"
        )

        # Save the image if a file path was provided, using high-quality settings for JPEG
        if file_path:
            if file_path.lower().endswith('.jpg') or file_path.lower().endswith('.jpeg'):
                self.out.save(file_path, 'JPEG', 100)
            else:
                self.out.save(file_path)  # PNG is lossless by default

        print(f'result saved: {file_path}')

    def import_example(self):
        idx = self.comboBox.currentIndex()
        img_path = self.example_paths[idx]
        pixmap = QPixmap(img_path)
        pixmap = pixmap.scaled(IMG_W, IMG_H,  Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

        self.canvas.clean_scene()
        self.canvas.setPhoto(pixmap)


    # Inference parameters __________________________________________
    """
    def define_ip_ref(self):
        try:
            img = QFileDialog.getOpenFileName(self, u"Ouverture de fichiers", "",
                                              "Image Files (*.png *.jpeg *.jpg *.bmp *.tif)")
            print(f'the following image will be loaded {img[0]}')
        except:
            pass

        if img[0] != '':
            # load and show new image
            self.ip_ref_img = img[0]
            self.ip_custom_path = img[0]

        self.change_inference_model()
        self.comboBox_ip_styles.setCurrentIndex(len(self.ip_img_paths))  # put combobox to 'custom'
    """

    def change_predefined_prompts(self):
        idx = self.comboBox.currentIndex()

        self.comboBox_style.clear()
        if idx < len(self.all_ip_paths):
            for style, img_path in zip(self.all_ip_styles[idx], self.all_ip_paths[idx]):
                icon = QIcon(img_path)
                self.comboBox_style.addItem(icon, style)

        self.textEdit.setText(self.all_ip_prompts[idx][0])
        self.update_image()

    def change_style(self):
        idx = self.comboBox.currentIndex()
        idx2 = self.comboBox_style.currentIndex()
        if self.checkBox_sp.isChecked():
            self.textEdit.setText(self.simple_prompts[idx])
        else:
            self.textEdit.setText(self.all_ip_prompts[idx][idx2])
        self.update_image()

    def update_brush_stroke(self):
        if self.checkBox.isChecked():
            self.update_image()

    """
    def change_preimg_style(self):
        i = self.comboBox_style.currentIndex()

        self.style = i
    """

    def update_image(self):
        # gather slider parameters:
        steps = self.step_slider.value()
        cfg = self.cfg_slider.value() / 10

        ip_strength = self.strength_slider.value() / 100

        # get prompts
        p = self.textEdit.toPlainText()
        np = self.textEdit_negative.toPlainText()

        # get ip
        idx = self.comboBox.currentIndex()
        ip_list = self.all_ip_paths[idx]

        idx2 = self.comboBox_style.currentIndex()
        ip_img_ref = ip_list[idx2]

        # get seed
        seed = int(self.lineEdit_seed.text())

        print(
            f'here are the parameters \n steps: {steps}\n cfg: {cfg}\n ipstrength: {ip_strength}\n prompt: {p}')

        print('capturing drawing')
        self.im = scene_to_image(self.canvas)

        # Check if image dimensions are 512x512
        if self.im.size != (IMG_W, IMG_H):
            print("Image dimensions are not good. Upscaling...")
            # Upscale the image to fit needed dimensions
            self.im = self.im.resize((IMG_W, IMG_H), Image.BICUBIC)

        self.im.save('input.png')

        # capture painted image
        print('running inference')
        self.out = self.infer(
            prompt=p,
            negative_prompt=np,
            image='input.png',
            num_inference_steps=steps,
            guidance_scale=cfg,
            seed=seed,
            ip_scale=ip_strength,
            ip_image_to_use=ip_img_ref
        )

        self.out.save('result.jpg')
        print('result saved')

        self.result_canvas.setPhoto(pixmap=QPixmap('result.jpg'))


def main(argv=None):
    """
    Creates the main window for the application and begins the \
    QApplication if necessary.

    :param      argv | [, ...] || None

    :return      error code
    """

    # Define installation path
    install_folder = os.path.dirname(__file__)

    app = None

    # create the application if necessary
    if (not QApplication.instance()):
        app = QApplication(argv)
        app.setStyle('Fusion')

        is_dark_theme = False
        print(f'Windows dark theme: {is_dark_theme}')

        if is_dark_theme:
            app.setStyleSheet("""
            QPushButton:checked {
                background-color: lightblue;
            }
            QPushButton:disabled {
                background-color: #666;
            }
            QWidget { background-color: #17161c; }
            QProgressBar {
                text-align: center;
                color: rgb(240, 240, 240);
                border-width: 1px; 
                border-radius: 10px;
                border-color: rgb(230, 230, 230);
                border-style: solid;
                background-color:rgb(207,207,207);
            }

            QProgressBar:chunk {
                background-color:rgb(50, 156, 179);
                border-radius: 10px;
            }
            """)

    # create the main window
    print('Launching the application')
    window = PaintLCM(is_dark_theme)
    window.show()

    # run the application if necessary
    if (app):
        return app.exec()

    # no errors since we're not running our own event loop
    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))
