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
import csv

# Params
BASE_DIR = res.find('img/AI_ref_images_bo')
IMG_W = 1024
IMG_H = 1024
CAPTURE_INT = 1000  # milliseconds
HD_RES = 1024  # resolution for upscale
SIMPLE_PROMPTS = ['A building architectural render',
                  'A building artistic architectural drawing',
                  'A city',
                  'The cross section of a building',
                  'A facade elevation',
                  'A floor plan',
                  'The drawing of an interior',
                  'Interior architectural render',
                  'Isometric building',
                  'Ground plan landscape architect'
                  ]

EXAMPLE_PROMPTS = ['black and white coloring book illustration of a building, white background, lineart, inkscape, simple lines',
                   'black and white coloring book illustration of a building, white background, lineart, inkscape, simple lines',
                   'black and white coloring book illustration of a city, white background, lineart, inkscape, simple lines',
                   'cross section of a building, children coloring book, white background, lineart, inkscape, simple lines',
                   'a building facade in a children coloring book, coloring page, lineart, white background',
                   'coloring page of a simple floor plan, lineart, orthographic, CAD',
                   'coloring page of an interior, line art, white background',
                   'coloring page of an interior, line art, white background',
                   'isometric building in a coloring book, line art, white background, simplistic',
                   'a site map, black and white, coloring book drawing, line art',
                   'some architectural drawing']

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

    # Render full image (without background)
    viewer._photo.setPixmap(viewer.coloring_layer)

    # Create a QPainter to render the scene into the QImage
    painter = QPainter(image)
    viewer.render(painter)
    painter.end()

    file_path = 'color_input.png'
    image.save(file_path)
    color_im = Image.open(file_path)

    # Render only black lines
    image_line = QImage(viewer.viewport().size(), QImage.Format.Format_ARGB32_Premultiplied)
    viewer._photo.setPixmap(viewer.drawing_layer)

    painter = QPainter(image_line)
    viewer.render(painter)
    painter.end()

    file_path = 'line_input.png'
    image_line.save(file_path)

    # Open the image and convert it to RGB mode
    pil_img = Image.open(file_path).convert('RGB')

    # Invert the lines
    inverted_img = ImageOps.invert(pil_img)

    # remove temp
    os.remove(file_path)

    # Save the inverted image
    inverted_file_path = 'inv_line_input.png'
    inverted_img.save(inverted_file_path)

    # set viewer again
    viewer.update_composite_pixmap()

    return inverted_img, color_im # lines, colors

class DrawingWindow(QMainWindow):
    make_bigger = pyqtSignal()
    make_smaller = pyqtSignal()
    # Define a custom signal called 'closed'
    closed = pyqtSignal()

    def __init__(self, main_window):
        super().__init__()
        basepath = os.path.dirname(__file__)
        basename = 'float_draw'
        uifile = os.path.join(basepath, '%s.ui' % basename)
        uic.loadUi(uifile, self)
        self.setWindowTitle("Drawing Window")

        # Store reference to main window
        self.main_window = main_window


        # Copy toolbar actions from the main window
        self.toolbar = self.addToolBar("Secondary Toolbar")
        self.action_list = ['Pencil','Brush','Eraser','Segm. Brush', 'Choose object','Start again!']
        self.action_in_group = ['Pencil','Brush','Eraser','Segm. Brush']
        self.copy_toolbar_actions(main_window)

        self.pushButton_plus.clicked.connect(self.make_dr_big)
        self.pushButton_min.clicked.connect(self.make_dr_small)

    def make_dr_big(self):
        self.make_bigger.emit()

    def make_dr_small(self):
        self.make_smaller.emit()
    def copy_toolbar_actions(self, main_window):
        action_group = QActionGroup(self)
        action_group.setExclusive(True)

        for action in main_window.toolBar.actions():
            if action.text() in self.action_list:
                if not action.text() in self.action_in_group:
                    new_action = QAction(action.icon(), action.text(), self)
                    new_action.triggered.connect(action.trigger)
                    self.toolbar.addAction(new_action)
                else:
                    new_action = QAction(action.icon(), action.text(), self, checkable=True)
                    new_action.triggered.connect(action.trigger)
                    self.toolbar.addAction(new_action)
                    action_group.addAction(new_action)

    def closeEvent(self, event):
        # Emit the 'closed' signal when the window is closed
        self.closed.emit()
        super().closeEvent(event)


class ColorDialog(QDialog):
    def __init__(self, parent=None):
        super(ColorDialog, self).__init__(parent)
        self.selected_color = None
        self.setWindowTitle("Select a Color")

        # Load the categories and colors from the CSV file
        self.colors = self.load_colors_from_csv('resources/other/out_categories.csv')

        # Create the grid layout
        layout = QGridLayout()

        # Create buttons and add them to the layout
        for i, (name, color) in enumerate(self.colors.items()):
            button = QPushButton()
            button.setIcon(QIcon(f'resources/img/icon/cat_{name}.png'))  # Assuming icons are named and located appropriately
            button.setIconSize(QPixmap(f'resources/img/icon/cat_{name}.png').size())
            button.clicked.connect(lambda checked, color=color: self.select_color(color))
            layout.addWidget(button, i // 3, i % 3)

        self.setLayout(layout)

    def load_colors_from_csv(self, file_path):
        colors = {}
        with open(file_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                color_code = tuple(map(int, row["Color_Code (R,G,B)"].strip("()").split(',')))
                name = row["Name"]
                colors[name] = color_code
        return colors

    def select_color(self, color):
        self.selected_color = color
        print(f'selected color: {color}')
        self.accept()

    def get_selected_color(self):
        return self.selected_color

class CustomDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enter Name and Prompt")

        self.name_label = QLabel("Name:")
        self.name_input = QLineEdit(self)

        self.prompt_label = QLabel("Prompt:")
        self.prompt_input = QTextEdit(self)

        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.name_label)
        self.layout.addWidget(self.name_input)
        self.layout.addWidget(self.prompt_label)
        self.layout.addWidget(self.prompt_input)
        self.layout.addWidget(self.ok_button)
        self.setLayout(self.layout)

    def getInputs(self):
        return self.name_input.text(), self.prompt_input.toPlainText()


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
        ag.addAction(self.spray_action)

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

        self.secondary_window = None

        # loads models
        self.models = model_list
        self.models_ids = model_ids

        # read custom models
        custom_models_file_list = os.listdir('custom_models')
        for i, file in enumerate(custom_models_file_list):
            if not file.endswith('txt'):
                self.models.append(f'custom{i}:{file}')
                self.models_ids.append(os.path.join('custom_models', file))

        self.model_id = self.models_ids[0]

        # initial parameters
        self.infer = load_models_multiple_cn()
        self.im = None
        self.original_parent = None
        self.gray_image = None
        self.resized_image = None
        self.actions_visible = True
        self.initialization = True
        # self.dockWidget_3.hide() # hide floating drawing zone

        # prepare sequence recording
        self.is_recording = False
        self.record_folder = ''
        self.n_frame = 0

        self.line_mode = 0

        # add capture box
        self.box = wid.TransparentBox(self.img_dim)
        self.capture_interval = CAPTURE_INT
        self.timer = QTimer(self)

        # pre-img parameters
        self.simple_prompts = SIMPLE_PROMPTS
        self.example_prompts = EXAMPLE_PROMPTS

        self.type_names = []  # To store the type names as read from folder names
        self.all_ip_styles = []  # To store style names
        self.all_ip_prompts = []  # To store corresponding prompts
        self.all_ip_paths = []  # To store image paths
        self.generate_ai_database()

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
        self.comboBox_model.addItems(model_list)

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
                                   self.import_action,
                                   self.sequence_action,
                                   self.process_action]

        # Create a snapshot of the toolbar with positions
        self.original_toolbar_state = [(action, self.toolBar.actions().index(action)) for action in
                                       self.toolBar.actions() if action in self.toggleable_actions]

        self.add_icon(res.find(f'img/icon/brush.png'), self.brush_action)
        self.add_icon(res.find(f'img/icon/brush_color.png'), self.spray_action)
        self.add_icon(res.find(f'img/icon/eraser.png'), self.eraser_action)
        self.add_icon(res.find(f'img/icon/pencil.png'), self.pencil_action)
        self.add_icon(res.find(f'img/icon/bezier.png'), self.bezier_action)
        self.add_icon(res.find(f'img/icon/mop.png'), self.reset_action)
        self.add_icon(res.find(f'img/icon/save.png'), self.export_action)
        self.add_icon(res.find(f'img/icon/hd.png'), self.exporthd_action)
        self.add_icon(res.find(f'img/icon/crop.png'), self.capture_action)
        self.add_icon(res.find(f'img/icon/image.png'), self.import_action)
        self.add_icon(res.find(f'img/icon/toggle.png'), self.toggle_action)
        self.add_icon(res.find(f'img/icon/movie.png'), self.sequence_action)
        self.add_icon(res.find(f'img/icon/reprocess.png'), self.process_action)
        self.add_icon(res.find(f'img/icon/palette.png'), self.palette_action)

        # create connections
        self.create_connections()
        self.change_type()

        # run first inference
        self.initialization = False
        self.update_image()

    # signals and connections ___________________________________
    def create_connections(self):
        # timer
        self.timer.timeout.connect(self.captureScreen)

        # actions
        self.brush_action.triggered.connect(self.switch_to_brush)
        self.spray_action.triggered.connect(self.switch_to_airbrush)
        self.eraser_action.triggered.connect(self.switch_to_eraser)
        self.pencil_action.triggered.connect(self.switch_to_pencil)
        self.bezier_action.triggered.connect(self.switch_to_bezier)
        self.reset_action.triggered.connect(self.reset_canvas)
        self.import_action.triggered.connect(self.import_image)
        self.export_action.triggered.connect(self.save_output)
        self.exporthd_action.triggered.connect(self.save_output_hd)
        self.capture_action.triggered.connect(self.toggle_capture)
        self.sequence_action.triggered.connect(self.record_sequence)
        self.process_action.triggered.connect(self.process_folder)
        self.palette_action.triggered.connect(self.choose_color)

        # pushbuttons
        # self.actionLoad_IP_Adapter_reference_image.triggered.connect(self.define_ip_ref)
        self.pushButton.clicked.connect(self.manual_update)
        self.pushButton_example.clicked.connect(self.generate_example)

        self.pushButton_import_style.clicked.connect(self.import_custom_style)

        # when editing canvas --> update inference
        self.canvas.endDrawing.connect(self.update_brush_stroke)

        # comboboxes
        self.comboBox.currentIndexChanged.connect(self.change_type)
        self.comboBox_style.currentIndexChanged.connect(self.change_style)
        self.comboBox_lines.currentIndexChanged.connect(self.change_capture_option)
        self.comboBox_model.currentIndexChanged.connect(self.change_model)

        # sliders
        self.step_slider.valueChanged.connect(self.update_image)
        self.cfg_slider.valueChanged.connect(self.update_image)
        self.strength_slider.valueChanged.connect(self.update_image)
        self.strength_slider_cn.valueChanged.connect(self.update_image)
        self.strength_slider_cn_2.valueChanged.connect(self.update_image)

        # checkboxes
        self.checkBox_sp.stateChanged.connect(self.change_style)
        self.checkBox_hide.stateChanged.connect(self.toggle_canvas)
        self.checkBox_float_draw.stateChanged.connect(self.toggle_secondary_window)

        # other actions
        self.actionAdvanced_options.triggered.connect(self.toggle_dock_visibility)

        # seed edit
        self.lineEdit_seed.textChanged.connect(self.update_image)

        self.toggle_action.triggered.connect(self.toggle_tools)

    # secondary drawing window
    def toggle_secondary_window(self):
        if self.checkBox_float_draw.isChecked():
            self.open_secondary_window()
        else:
            self.secondary_window.close()
            self.close_secondary_window()

    def open_secondary_window(self):
        if self.secondary_window is None:
            self.secondary_window = DrawingWindow(self)

        self.canvas.setParent(self.secondary_window)
        self.secondary_window.horizontalLayout.addWidget(self.canvas)
        self.secondary_window.closed.connect(self.close_secondary_window)
        self.secondary_window.make_bigger.connect(lambda: self.scale_scene('plus'))
        self.secondary_window.make_smaller.connect(lambda: self.scale_scene('min'))
        self.secondary_window.show()

    def close_secondary_window(self):
        if self.checkBox_float_draw.isChecked():
            self.checkBox_float_draw.setChecked(False)

        # add drawing zone to principal window
        self.horizontalLayout_4.removeWidget(self.result_canvas)
        self.canvas.setParent(self)
        self.horizontalLayout_4.addWidget(self.canvas)
        self.horizontalLayout_4.addWidget(self.result_canvas)


    # drawing functions _________________________________________
    def choose_color_old(self):
        self.canvas.set_color()


    def choose_color(self):
        dialog = ColorDialog()
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_color = dialog.get_selected_color()

        self.canvas.current_color = QColor(*selected_color)


    def switch_to_pencil(self):
        self.palette_action.setEnabled(False)
        self.canvas.terminate_bezier()
        self.canvas.brush_size = 3
        self.canvas.brush_cur = self.canvas.create_circle_cursor(3)
        self.canvas.set_tool('pencil')
        self.canvas.active_layer = 0

    def switch_to_airbrush(self):
        # activate color palette

        self.palette_action.setEnabled(True)
        self.canvas.terminate_bezier()
        self.canvas.brush_size = 15
        self.canvas.brush_cur = self.canvas.create_circle_cursor(15)
        self.canvas.set_tool('airbrush')
        self.canvas.active_layer = 1 # activate coloring layer

        self.choose_color()


    def switch_to_bezier(self):
        self.palette_action.setEnabled(False)
        self.canvas.set_tool('bezier')
        self.canvas.active_layer = 0

    def switch_to_brush(self):
        self.palette_action.setEnabled(False)
        self.canvas.terminate_bezier()
        self.canvas.brush_size = 10
        self.canvas.brush_cur = self.canvas.create_circle_cursor(10)
        self.canvas.set_tool('brush')
        self.canvas.active_layer = 0

    def switch_to_eraser(self):
        self.palette_action.setEnabled(False)
        self.canvas.terminate_bezier()
        self.canvas.brush_size = 20
        self.canvas.set_tool('eraser')
        self.canvas.brush_cur = self.canvas.create_circle_cursor(20)
        self.canvas.change_to_brush_cursor()

    # general functions __________________________________________

    def generate_ai_database(self):
        base_dir = BASE_DIR

        # reset data
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

    def record_sequence(self):
        if self.sequence_action.isChecked():
            self.process_action.setEnabled(False)
            # change flag
            self.is_recording = True

            # let the user choose an output folder
            out_dir = str(QFileDialog.getExistingDirectory(self, "Select output_folder"))
            while not os.path.isdir(out_dir):
                QMessageBox.warning(self, "Warning",
                                    "Oops! Not a folder!")
                out_dir = str(QFileDialog.getExistingDirectory(self, "Select output_folder"))

            self.record_folder = out_dir
            self.inf_folder = os.path.join(self.record_folder, 'inference')
            self.input_line_folder = os.path.join(self.record_folder, 'inputs_lines')
            self.input_seg_folder = os.path.join(self.record_folder, 'inputs_seg')
            # create the new subfolders to save frames
            new_dir(self.inf_folder)
            new_dir(self.input_line_folder)
            new_dir(self.input_seg_folder)

        else:
            self.process_action.setEnabled(True)
            # change flag
            self.is_recording = False
            self.compile_video()
            self.n_frame = 0

    def compile_video(self):
        path_inference = os.path.join(self.inf_folder, 'inference_video.mp4')
        path_line_input = os.path.join(self.input_line_folder, 'input_line_video.mp4')
        path_seg_input = os.path.join(self.input_seg_folder, 'input_seg_video.mp4')
        lcm.create_video(self.inf_folder, path_inference, 3)
        lcm.create_video(self.input_line_folder, path_line_input, 3)
        lcm.create_video(self.input_seg_folder, path_seg_input, 3)

    def scale_scene(self, direction):
        # Get the current scene rect
        rect = self.canvas.sceneRect()
        w, h = int(rect.width()), int(rect.height())  # Ensure w and h are integers

        # Determine the scaling factor
        if direction == 'plus':
            factor = 1.2
        elif direction == 'min':
            factor = 0.8
        else:
            return  # Invalid direction

        # Scale the dimensions
        new_w, new_h = int(w * factor), int(h * factor)

        self.canvas.scale_canvas(new_w, new_h)

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
            self.toggle_push_buttons()
        else:
            # Restore each action at its original position
            for action, position in sorted(self.original_toolbar_state, key=lambda x: x[1]):
                before_action = self.toolBar.actions()[position] if len(self.toolBar.actions()) > position else None
                self.toolBar.insertAction(before_action, action)
            self.actions_visible = True
            self.toggle_action.setText("Reduce toolset")
            self.status_bar.showMessage("Full Toolbar")
            self.toggle_push_buttons()

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
            self.pushButton_import_style.hide()
        else:
            self.pushButton_example.show()
            self.pushButton_import_style.show()

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

    def generate_example(self):
        # create blank image
        white_image = Image.new("RGB", (self.img_dim[0], self.img_dim[1]), "white")
        # load prompt
        idx = self.comboBox.currentIndex()
        prompt = self.example_prompts[idx]
        negative_prompt = 'realistic, colors, detailed, writing, text'
        out = self.infer(
            prompt,
            negative_prompt,
            [white_image, white_image],
            num_inference_steps=7,
            guidance_scale=0.7,
            strength=0.9,
            seed=random.randrange(0, 2 ** 63),
            ip_scale=0.2,
            ip_image_to_use=res.find('img/examples/city_default.png'),
            cn_strength=[0, 0],
        )

        # Convert the RGB image to grayscale
        grayscale_image = out.convert("L")
        optimal_threshold = otsu_threshold(grayscale_image)
        print(f"Optimal threshold: {optimal_threshold}")

        # Convert the grayscale image to binary (black and white) using a threshold
        threshold = optimal_threshold  # You can adjust the threshold as needed
        binary_image = grayscale_image.point(lambda x: 0 if x < threshold else 255, '1')
        binary_image.save('example.png')
        # save pixmap in drawing zone
        self.canvas.set_image_in_drawing(QPixmap("example.png"))

        # update render
        self.update_image()

    def find_custom_index(self):
        custom_text = "user"
        for index in range(self.comboBox.count()):
            if custom_text in self.comboBox.itemText(index):
                return index
        return -1

    def import_custom_style(self):
        # file selection dialog
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image File", "",
                                                   "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)")
        if file_name:
            # check max dimensions
            image = cv2.imread(file_name)
            max_dimension = 1024
            height, width = image.shape[:2]
            scaling_factor = max_dimension / max(height, width)

            if scaling_factor < 1:
                new_size = (int(width * scaling_factor), int(height * scaling_factor))
                resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
            else:
                resized_image = image

            # open diaglog to get name and prompt
            dialog = CustomDialog()
            if dialog.exec() == QDialog.DialogCode.Accepted:
                name, prompt = dialog.getInputs()

                if name and prompt:
                    # save image and text to database
                    image_name = f"{name}.png"
                    dest_path = os.path.join(BASE_DIR, 'user', image_name)
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    cv2.imwrite(dest_path, resized_image)

                    text_file_path = os.path.join(BASE_DIR, 'user', f"{name}.txt")
                    with open(text_file_path, 'w') as text_file:
                        text_file.write(prompt)

                    self.initialization = True
                    # update data
                    self.generate_ai_database()

                    # reset type combobox
                    self.comboBox.clear()
                    self.comboBox.addItems([f'Type {i + 1} ({name})' for i, name in enumerate(self.type_names)])

                    # update_styles
                    self.change_type()

                    # Find the index of 'custom' in the comboBox
                    index = self.find_custom_index()
                    print(index)
                    self.comboBox.setCurrentIndex(index)

                    self.initialization = False
                    self.update_image()

                else:
                    QMessageBox.warning(self, "Input Error", "Name and prompt cannot be empty.")
            else:
                QMessageBox.warning(self, "Dialog Canceled", "Operation was canceled.")

    def import_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image File", "",
                                                   "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)")
        if file_name:
            # Resize image to maximum dimension
            image = cv2.imread(file_name)
            max_dimension = max(self.img_dim[0], self.img_dim[1])
            height, width = image.shape[:2]
            scaling_factor = max_dimension / max(height, width)

            if scaling_factor < 1:
                new_size = (int(width * scaling_factor), int(height * scaling_factor))
                resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
            else:
                resized_image = image

            # Compile new size parameters
            self.img_dim = (int(width * scaling_factor), int(height * scaling_factor))

            self.result_canvas.create_new_scene(self.img_dim[0], self.img_dim[1])
            self.canvas.create_new_scene(self.img_dim[0], self.img_dim[1])

            # Update capture window
            self.box = wid.TransparentBox(self.img_dim)

            # Convert to grayscale
            self.resized_image = resized_image
            self.temp_file_path = 'temp_resized_image.png'
            cv2.imwrite(self.temp_file_path, resized_image)

            # Create a message box
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Icon.Question)
            msg_box.setWindowTitle('Import Image')
            msg_box.setText('How would you like to import the image?')

            # Add buttons
            support_button = msg_box.addButton('As support for drawing', QMessageBox.ButtonRole.YesRole)
            line_button = msg_box.addButton('For lines layer', QMessageBox.ButtonRole.NoRole)
            segmentation_button = msg_box.addButton('For segmentation layer', QMessageBox.ButtonRole.HelpRole)
            segmentation_lines_button = msg_box.addButton('For segmentation and line layers', QMessageBox.ButtonRole.HelpRole)

            # Execute the message box and get the user's choice
            msg_box.exec()

            # Call the appropriate method based on the user's choice
            if msg_box.clickedButton() == support_button:
                self.import_support_image()
            elif msg_box.clickedButton() == line_button:
                self.import_line_image()
            elif msg_box.clickedButton() == segmentation_button:
                self.import_segmentation_image()
            elif msg_box.clickedButton() == segmentation_lines_button:
                self.import_segmentation_lines_image()

    def import_support_image(self):
        image = cv2.imread(self.temp_file_path, cv2.IMREAD_UNCHANGED)

        # Check if the image has an alpha channel
        if image.shape[2] == 3:
            # Add an alpha channel
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

        # Reduce opacity to 50%
        alpha_channel = image[:, :, 3]
        alpha_channel = (alpha_channel * 0.5).astype(np.uint8)
        image[:, :, 3] = alpha_channel

        # Save the image with reduced opacity
        cv2.imwrite(self.temp_file_path, image)

        # Convert back to QPixmap to display in a QLabel or other widget
        result_pixmap = QPixmap(self.temp_file_path)
        self.canvas.setPhoto(result_pixmap)

        self.canvas.has_background = True

    def import_line_image(self, update_inf = True):
        # line operation
        processed_image = lcm.screen_to_lines(self.resized_image, self.line_mode)

        # Convert the inverted image back to QPixmap
        final_pixmap = self.convertMatToQPixmap(processed_image)

        # save pixmap in drawing zone
        self.canvas.set_image_in_drawing(final_pixmap)

        # update render
        if update_inf:
            self.update_image()

    def import_segmentation_image(self):
        # line operation
        processed_image = lcm.img_to_seg(self.resized_image)
        # Convert the color segmentation array to a PIL image
        pil_image = Image.fromarray(processed_image)

        # Save the PIL image to the specified path
        pil_image.save('seg_result.png')

        # Convert the inverted image back to QPixmap
        final_pixmap = QPixmap('seg_result.png')

        # save pixmap in drawing zone
        self.canvas.set_image_in_color_layer(final_pixmap)

        # update render
        self.update_image()

    def import_segmentation_lines_image(self):
        self.import_line_image()
        self.import_segmentation_image()

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
            self.import_line_image(update_inf = False)

        # Should it update continuously
        if self.checkBox.isChecked():
            print('update from screen capture')
            self.update_image()

    def reprocess_capture(self, option):
        processed_image = lcm.screen_to_lines(self.resized_image, option)

        # Convert the inverted image back to QPixmap
        final_pixmap = self.convertMatToQPixmap(processed_image)

        # recompose drawing
        self.canvas.remove_stored_pixmap()

        # Set the processed image on the canvas
        self.canvas.set_image_in_drawing(final_pixmap)
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
    def change_model(self):
        idx = self.comboBox_model.currentIndex()
        self.model_id = self.models_ids[idx]
        print(f'chosen model:{self.model_id}')

        # Attempt to free up memory by explicitly deleting the previous model and calling garbage collector
        if hasattr(self, 'infer'):
            del self.infer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.infer = load_models_multiple_cn(model_id=self.model_id)
        self.update_image()

    def change_type(self):
        self.load_style_combobox()
        self.change_predefined_prompts()
        self.update_image()

    def load_style_combobox(self):
        change = False
        if not self.initialization:
            change = True
            self.initialization = True

        idx = self.comboBox.currentIndex()
        self.comboBox_style.clear()
        if idx < len(self.all_ip_paths):
            for style, img_path in zip(self.all_ip_styles[idx], self.all_ip_paths[idx]):
                icon = QIcon(img_path)
                self.comboBox_style.addItem(icon, style)

        if change:
            self.initialization = False

    def change_predefined_prompts(self):
        idx = self.comboBox.currentIndex()
        if not self.checkBox_keep_p.isChecked():
            self.textEdit.setText(self.all_ip_prompts[idx][0])

    def change_style(self):
        idx = self.comboBox.currentIndex()
        idx2 = self.comboBox_style.currentIndex()

        if not self.checkBox_keep_p.isChecked():
            if self.checkBox_sp.isChecked():
                self.textEdit.setText(self.simple_prompts[idx])
            else:
                self.textEdit.setText(self.all_ip_prompts[idx][idx2])

        self.update_image()

    def update_brush_stroke(self):
        if self.checkBox.isChecked():
            print('update from brush stroke')
            self.update_image()

    def process_folder(self):
        in_dir = str(QFileDialog.getExistingDirectory(self, "Select input folder"))
        while not os.path.isdir(in_dir):
            QMessageBox.warning(self, "Warning",
                                "Oops! Not a folder!")
            in_dir = str(QFileDialog.getExistingDirectory(self, "Select output_folder"))

        # output subfolder
        existing_folders = len([name for name in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir, name))])
        sub_dir = f"process{existing_folders}"
        out_sub = os.path.join(in_dir, sub_dir)
        new_dir(out_sub)

        # list image files
        file_list = os.listdir(in_dir)
        img_list = []

        for file in file_list:
            if file.endswith('.png'):
                img_list.append(os.path.join(in_dir,file))

        print(img_list)

        # process all images
        for i, img in enumerate(img_list):
            out_path = os.path.join(out_sub, f'out{i}.png')
            self.update_image(show_output=False, save_path=out_path,get_image_from_canvas=False,input_img_path=img)

    def manual_update(self):
        self.update_image()

    def update_image(self,
                     show_output=True,
                     save_path='result.png',
                     get_image_from_canvas=True,
                     input_img_path=''):

        if not self.initialization:
            # gather slider parameters:
            steps = self.step_slider.value()
            cfg = self.cfg_slider.value() / 10

            ip_strength = self.strength_slider.value() / 100
            cn_strength = self.strength_slider_cn.value() / 100
            cn_strength_2 = self.strength_slider_cn_2.value() / 100
            cn_strengths = [cn_strength, cn_strength_2]

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

            if get_image_from_canvas:
                print('capturing drawing')
                self.im, self.color_im = scene_to_image(self.canvas)

                # Check if image dimensions are correct
                if self.im.size != (self.img_dim[0], self.img_dim[1]):
                    print("Image dimensions are not good. Rescaling...")
                    # Upscale the image to fit needed dimensions
                    self.im = self.im.resize((self.img_dim[0], self.img_dim[1]), Image.BICUBIC)
                    self.color_im = self.color_im.resize((self.img_dim[0], self.img_dim[1]), Image.BICUBIC)

                self.im.save('inv_line_input.png')
                self.color_im.save('color_input.png')
                input_img_path = 'inv_line_input.png'
                input_color_path = 'color_input.png' # to use for segmentation

            # capture painted image
            print('running inference')
            """
            self.out = self.infer(
                prompt=self.p,
                negative_prompt=np,
                image=input_color_path,
                cn_image=input_img_path,
                num_inference_steps=steps,
                guidance_scale=cfg,
                seed=seed,
                ip_scale=ip_strength,
                ip_image_to_use=ip_img_ref,
                cn_strength=cn_strength
            )
            """
            self.out = self.infer(
                prompt=self.p,
                negative_prompt=np,
                images=[Image.open(input_img_path), Image.open(input_color_path)],
                num_inference_steps=steps,
                guidance_scale=cfg,
                seed=seed,
                ip_scale=ip_strength,
                ip_image_to_use=ip_img_ref,
                cn_strength=cn_strengths
            )

            self.out.save(save_path)
            print('result saved')
            if show_output:
                self.result_canvas.setPhoto(pixmap=QPixmap('result.png'))

            # save images if recording flag
            if self.is_recording:
                self.n_frame += 1
                frame_path = f"frame_{self.n_frame:04}.png"
                self.out.save(os.path.join(self.inf_folder, frame_path))
                self.im.save(os.path.join(self.input_line_folder, frame_path))
                self.color_im.save((os.path.join(self.input_seg_folder, frame_path)))


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
