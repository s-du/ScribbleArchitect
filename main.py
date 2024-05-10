"""
Scribble Architect allows to transforms simple doodles into architectural works!
Author: Samuel Dubois
Any remark/suggestion: sdu@bbri.be
"""

from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from PyQt6 import uic

import numpy as np

import lcm
import widgets as wid
import resources as res
from lcm import *
from PIL import Image, ImageOps

import torch
import os
import gc

# Params
BASE_DIR = res.find('img/AI_ref_images_bo')
LINE_METHODS = ['Sobel + BIL', 'Sobel Custom', 'Canny', 'Canny + L2', 'Canny + BIL', 'Canny + Blur', 'Random Forests', 'RF Custom',  'No processing']
IMG_W = 512
IMG_H = 512
CAPTURE_INT = 1000  # milliseconds
HD_RES = 1024 # resolution for upscale


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

        # add actions to action group (mutually exculive)
        ag = QActionGroup(self)
        ag.setExclusive(True)
        ag.addAction(self.brush_action)
        ag.addAction(self.eraser_action)
        ag.addAction(self.pencil_action)
        ag.addAction(self.bezier_action)

        self.img_dim = (IMG_W, IMG_H)
        self.canvas = wid.Canvas(self.img_dim)
        self.horizontalLayout_4.addWidget(self.canvas)

        self.result_canvas = wid.simpleCanvas(self.img_dim)
        self.horizontalLayout_4.addWidget(self.result_canvas)

        # status bar
        # Create and set status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Full Toolbar")

        # loads models
        self.models = model_list
        self.models_ids = model_ids

        # initial parameters
        self.infer = load_models()
        self.im = None
        self.original_parent = None
        self.gray_image = None
        self.resized_image = None
        self.actions_visible = True
        self.initialization = True

        self.line_mode = 0

        # add capture box
        self.box = wid.TransparentBox(self.img_dim)
        self.capture_interval = CAPTURE_INT
        self.timer = QTimer(self)

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

        base_dir = BASE_DIR
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

        # specific variables
        if not path.exists(cache_path):
            os.makedirs(cache_path, exist_ok=True)

        # seed value
        # Assuming a maximum value of 9999 for demonstration; adjust as needed
        validator = QIntValidator(0, 9999, self)

        # Apply the validator to the line edit
        self.lineEdit_seed.setValidator(validator)

        # comboboxes ---------------------------------------
        self.comboBox_lines.addItems(LINE_METHODS)
        self.comboBox.addItems([f'Type {i + 1} ({name})' for i, name in enumerate(self.type_names)])

        # After initializing the comboBox and adding items with icons, set the icon size
        desired_icon_width = 80  # Adjust the width as needed
        desired_icon_height = 80  # Adjust the height as needed

        self.comboBox_style.setIconSize(QSize(desired_icon_width, desired_icon_height))

        # ----------------------------------------------
        # Set textedits
        self.textEdit.setWordWrapMode(QTextOption.WrapMode.WordWrap)
        self.textEdit.setText(self.all_ip_prompts[0][0])
        self.textEdit_negative.setWordWrapMode(QTextOption.WrapMode.WordWrap)

        # toggle actions
        # add actions to complex toolbars
        self.toggleable_actions = [self.bezier_action,
                                   self.capture_action,
                                   self.export_action,
                                   self.exporthd_action,
                                   self.import_action]

        # Create a snapshot of the toolbar with positions
        self.original_toolbar_state = [(action, self.toolBar.actions().index(action)) for action in
                                       self.toolBar.actions() if action in self.toggleable_actions]

        if is_dark_theme:
            suf = '_white_tint'
            suf2 = '_white'
        else:
            suf = ''

        self.add_icon(res.find(f'img/brush{suf}.png'), self.brush_action)
        self.add_icon(res.find(f'img/eraser{suf}.png'), self.eraser_action)
        self.add_icon(res.find(f'img/pencil{suf}.png'), self.pencil_action)
        self.add_icon(res.find(f'img/bezier{suf}.png'), self.bezier_action)
        self.add_icon(res.find(f'img/mop{suf}.png'), self.color_action)
        self.add_icon(res.find(f'img/save_as{suf}.png'), self.export_action)
        self.add_icon(res.find(f'img/hd{suf}.png'), self.exporthd_action)
        self.add_icon(res.find(f'img/crop{suf}.png'), self.capture_action)
        self.add_icon(res.find(f'img/add{suf}.png'), self.import_action)
        self.add_icon(res.find(f'img/switch{suf}.png'), self.toggle_action)

        # create connections
        self.create_connections()

        self.change_predefined_prompts()

        # run first inference
        self.initialization = False
        self.update_image()

    # signals and connections ___________________________________
    def create_connections(self):
        # timer
        self.timer.timeout.connect(self.captureScreen)

        # actions
        self.brush_action.triggered.connect(self.switch_to_brush)
        self.eraser_action.triggered.connect(self.switch_to_eraser)
        self.pencil_action.triggered.connect(self.switch_to_pencil)
        self.bezier_action.triggered.connect(self.switch_to_bezier)
        self.color_action.triggered.connect(self.reset_canvas)
        self.import_action.triggered.connect(self.import_image)
        self.export_action.triggered.connect(self.save_output)
        self.exporthd_action.triggered.connect(self.save_output_hd)
        self.capture_action.triggered.connect(self.toggle_capture)

        # pushbuttons
        # self.actionLoad_IP_Adapter_reference_image.triggered.connect(self.define_ip_ref)
        self.pushButton.clicked.connect(self.update_image)
        self.pushButton_example.clicked.connect(self.import_example)

        # when editing canvas --> update inference
        self.canvas.endDrawing.connect(self.update_brush_stroke)

        # comboboxes
        self.comboBox.currentIndexChanged.connect(self.change_predefined_prompts)
        self.comboBox_style.currentIndexChanged.connect(self.change_style)
        self.comboBox_lines.currentIndexChanged.connect(self.change_capture_option)

        # sliders
        self.step_slider.valueChanged.connect(self.update_image)
        self.cfg_slider.valueChanged.connect(self.update_image)
        self.strength_slider.valueChanged.connect(self.update_image)
        self.strength_slider_cn.valueChanged.connect(self.update_image)

        # checkboxes
        self.checkBox_sp.stateChanged.connect(self.change_style)
        self.checkBox_hide.stateChanged.connect(self.toggle_canvas)

        # other actions
        self.actionAdvanced_options.triggered.connect(self.toggle_dock_visibility)

        # seed edit
        self.lineEdit_seed.textChanged.connect(self.update_image)

        self.toggle_action.triggered.connect(self.toggle_tools)

    # drawing functions _________________________________________
    def switch_to_pencil(self):
        self.canvas.terminate_bezier()
        self.canvas.brush_size = 3
        self.canvas.brush_cur = self.canvas.create_circle_cursor(3)
        self.canvas.set_tool('brush')

    def switch_to_bezier(self):
        self.canvas.set_tool('bezier')

    def switch_to_brush(self):
        self.canvas.terminate_bezier()
        self.canvas.brush_size = 10
        self.canvas.brush_cur = self.canvas.create_circle_cursor(10)
        self.canvas.set_tool('brush')

    def switch_to_eraser(self):
        self.canvas.terminate_bezier()
        self.canvas.brush_size = 20
        self.canvas.set_tool('eraser')
        self.canvas.brush_cur = self.canvas.create_circle_cursor(20)
        self.canvas.change_to_brush_cursor()

    # general functions __________________________________________

    def snapshot_toolbar(self):
        """ Capture the current state of the toolbar. """
        return self.toolBar.actions()

    def toggle_tools(self):
        """Toggle the visibility of selected actions, preserving their order."""
        if self.actions_visible:
            for action, position in self.original_toolbar_state:
                self.toolBar.removeAction(action)
            self.actions_visible = False
            self.toggle_action.setText("Expand toolset")
            self.status_bar.showMessage("Reduced Toolbar")
        else:
            # Restore each action at its original position
            for action, position in sorted(self.original_toolbar_state, key=lambda x: x[1]):
                before_action = self.toolBar.actions()[position] if len(self.toolBar.actions()) > position else None
                self.toolBar.insertAction(before_action, action)
            self.actions_visible = True
            self.toggle_action.setText("Reduce toolset")
            self.status_bar.showMessage("Full Toolbar")

        self.toggle_dock_visibility()
        # self.toggle_push_buttons()

    def toggle_dock_visibility(self):
        if self.dockWidget_2.isVisible():
            self.dockWidget_2.hide()
        else:
            self.dockWidget_2.show()

    def toggle_push_buttons(self):
        if self.pushButton_example.isVisible():
            self.pushButton_example.hide()
        else:
            self.pushButton_example.show()

    def toggle_canvas(self):
        # Hide or show canvas based on checkbox state
        if self.checkBox_hide.isChecked():
            self.canvas.hide()
        else:
            self.canvas.show()

        # Adjust the size of the window
        self.adjustSize()

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

    def save_output_hd(self):
        # add code for file dialog
        file_path, _ = QFileDialog.getSaveFileName(
            None, "Save Image", "", "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg *.JPEG)"
        )

        print('running upscale...')
        out = lcm.tile_upscale(self.out, self.p, HD_RES)

        # Save the image if a file path was provided, using high-quality settings for JPEG
        if file_path:
            if file_path.lower().endswith('.jpg') or file_path.lower().endswith('.jpeg'):
                out.save(file_path, 'JPEG', 100)
            else:
                out.save(file_path)  # PNG is lossless by default

        print(f'result saved: {file_path}')

    def import_example(self):
        idx = self.comboBox.currentIndex()
        img_path = self.example_paths[idx]
        pixmap = QPixmap(img_path)
        pixmap = pixmap.scaled(self.img_dim[0], self.img_dim[1], Qt.AspectRatioMode.KeepAspectRatio,
                               Qt.TransformationMode.SmoothTransformation)

        self.canvas.clean_scene()
        self.canvas.setPhoto(pixmap)

    def import_image(self):
        # file selection dialog
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image File", "",
                                                   "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)")
        if file_name:
            # resize image to maximum dimension
            image = cv2.imread(file_name)
            max_dimension = max(self.img_dim[0], self.img_dim[1])
            height, width = image.shape[:2]
            scaling_factor = max_dimension / max(height, width)
            if scaling_factor < 1:
                new_size = (int(width * scaling_factor), int(height * scaling_factor))
                resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
            else:
                resized_image = image

            # compile new size parameters

            self.img_dim = (int(width * scaling_factor), int(height * scaling_factor))

            self.result_canvas.create_new_scene(self.img_dim[0], self.img_dim[1])
            self.canvas.create_new_scene(self.img_dim[0], self.img_dim[1])

            # update capture window
            self.box = wid.TransparentBox(self.img_dim)

            # convert to grayscale
            self.resized_image = resized_image

            # line operation
            processed_image = lcm.screen_to_lines(self.resized_image, self.line_mode)

            # Convert the inverted image back to QPixmap
            final_pixmap = self.convertMatToQPixmap(processed_image)

            # save pixmap in drawing zone
            self.canvas.setPhoto(final_pixmap)

            # update render
            self.update_image()

    # Screen capture __________________________________________
    def toggle_capture(self):
        if self.capture_action.isChecked():
            # disable tools
            self.brush_action.setEnabled(False)
            self.eraser_action.setEnabled(False)

            # remove existing items
            self.canvas.clear_drawing()

            # launch capture
            self.box.show()
            self.timer.start(self.capture_interval)

        else:
            self.brush_action.setEnabled(True)
            self.eraser_action.setEnabled(True)

            self.timer.stop()
            # stop capture
            self.box.hide()

    def captureScreen(self):
        # Get geometry of the transparent box
        x, y, width, height = self.box.geometry().getRect()
        print(width, height)

        screen = QApplication.primaryScreen()
        if screen is not None:
            pixmap = screen.grabWindow(0, x + 6, y + 6, width - 12, height - 12)

            # Convert QPixmap to QImage
            qimage = pixmap.toImage()

            # Convert QImage to OpenCV format
            temp_image = self.convertQImageToMat(qimage)

            # Convert to grayscale
            self.resized_image = temp_image

            # convert to edge image
            processed_image = lcm.screen_to_lines(temp_image, self.line_mode)

            # Convert the inverted image back to QPixmap
            final_pixmap = self.convertMatToQPixmap(processed_image)

            # Set the processed image on the canvas
            self.canvas.setPhoto(final_pixmap)

        # Should it update continuously
        if self.checkBox.isChecked():
            print('update from screen capture')
            self.update_image()

    def reprocess_capture(self, option):
        processed_image = lcm.screen_to_lines(self.resized_image, option)

        # Convert the inverted image back to QPixmap
        final_pixmap = self.convertMatToQPixmap(processed_image)

        # Set the processed image on the canvas
        self.canvas.setPhoto(final_pixmap)
        self.update_image()

    def change_capture_option(self):
        idx = self.comboBox_lines.currentIndex()
        self.line_mode = idx
        if self.resized_image is not None:
            self.reprocess_capture(idx)

    def convertQImageToMat(self, incomingImage):
        ''' Convert QImage to OpenCV format '''
        incomingImage = incomingImage.convertToFormat(QImage.Format.Format_RGB32)
        width = incomingImage.width()
        height = incomingImage.height()

        ptr = incomingImage.bits()
        # Adjust byte count calculation based on PyQt version
        if hasattr(incomingImage, 'sizeInBytes'):  # PyQt 5.10 and newer
            ptr.setsize(incomingImage.sizeInBytes())
        else:  # PyQt versions older than 5.10
            ptr.setsize(height * width * 4)  # 4 bytes per pixel in RGB32

        arr = np.array(ptr).reshape(height, width, 4)  # Copies the data
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

    def convertMatToQPixmap(self, mat):
        ''' Convert OpenCV image format to QPixmap '''
        rgb_image = cv2.cvtColor(mat, cv2.COLOR_GRAY2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(w, h)
        return QPixmap.fromImage(p)

    def closeEvent(self, event):
        # Explicitly close the transparent box when the main window is closed
        self.box.close()
        event.accept()

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
        if not self.checkBox_keep_p.isChecked():
            self.textEdit.setText(self.all_ip_prompts[idx][0])
            print('update from change of predefined prompt')
        self.update_image()

    def change_style(self):
        idx = self.comboBox.currentIndex()
        idx2 = self.comboBox_style.currentIndex()

        if not self.checkBox_keep_p.isChecked():
            if self.checkBox_sp.isChecked():
                self.textEdit.setText(self.simple_prompts[idx])
            else:
                self.textEdit.setText(self.all_ip_prompts[idx][idx2])

        print('update from change of style')
        self.update_image()

    def update_brush_stroke(self):
        if self.checkBox.isChecked():
            print('update from brush stroke')
            self.update_image()

    def update_image(self):
        if not self.initialization:
            # gather slider parameters:
            steps = self.step_slider.value()
            cfg = self.cfg_slider.value() / 10

            ip_strength = self.strength_slider.value() / 100
            cn_strength = self.strength_slider_cn.value() / 100

            # get prompts
            self.p = self.textEdit.toPlainText()
            np = self.textEdit_negative.toPlainText()

            # get ip
            idx = self.comboBox.currentIndex()
            ip_list = self.all_ip_paths[idx]

            idx2 = self.comboBox_style.currentIndex()
            ip_img_ref = ip_list[idx2]

            # get seed
            seed = int(self.lineEdit_seed.text())

            print(
                f'here are the parameters \n steps: {steps}\n cfg: {cfg}\n ipstrength: {ip_strength}\n prompt: {self.p}')

            print('capturing drawing')
            self.im = scene_to_image(self.canvas)

            # Check if image dimensions are correct
            if self.im.size != (self.img_dim[0], self.img_dim[1]):
                print("Image dimensions are not good. Upscaling...")
                # Upscale the image to fit needed dimensions
                self.im = self.im.resize((self.img_dim[0], self.img_dim[1]), Image.BICUBIC)

            self.im.save('input.png')

            # capture painted image
            print('running inference')
            self.out = self.infer(
                prompt=self.p,
                negative_prompt=np,
                image='input.png',
                num_inference_steps=steps,
                guidance_scale=cfg,
                seed=seed,
                ip_scale=ip_strength,
                ip_image_to_use=ip_img_ref,
                cn_strength=cn_strength
            )

            self.out.save('result.png')
            print('result saved')

            self.result_canvas.setPhoto(pixmap=QPixmap('result.png'))


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
    window.showMaximized()

    # run the application if necessary
    if (app):
        return app.exec()

    # no errors since we're not running our own event loop
    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))
