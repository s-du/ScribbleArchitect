"""
Author: SDU (sdu@bbri.be)
"""
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

import numpy as np
from scipy.interpolate import UnivariateSpline


class TransparentBox(QWidget):
    def __init__(self, size):
        super().__init__()
        self.w, self.h = size
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.dragging = False
        self.setGeometry(100, 100, self.w + 12, self.h + 12)  # Initial position and size

    def paintEvent(self, event):
        painter = QPainter(self)
        pen = QPen(QColor(0, 120, 215), 6)  # Increased pen width to 6px
        painter.setPen(pen)
        painter.drawRect(0, 0, self.width(), self.height())  # Draw box edges

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = True
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if self.dragging and event.buttons() == Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self.drag_position)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False

    def enterEvent(self, event):
        self.setCursor(Qt.CursorShape.SizeAllCursor)

    def leaveEvent(self, event):
        self.unsetCursor()


class simpleCanvas(QGraphicsView):
    def __init__(self, img_size):
        super().__init__()

        self.w, self.h = img_size

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setSceneRect(0, 0, self.w, self.h)
        self.setMinimumSize(self.w, self.h)
        self.setMaximumSize(self.w, self.h)

        self._photo = QGraphicsPixmapItem()
        self.scene.addItem(self._photo)
        self.pixmap = None

        self.setBackgroundBrush(QBrush(QColor(180, 180, 180)))
        self.setContentsMargins(0, 0, 0, 0)
        self.setViewportMargins(0, 0, 0, 0)

        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

    def create_new_scene(self, w, h):
        self.w = w
        self.h = h
        self.scene.clear()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setSceneRect(0, 0, w, h)
        self.setMinimumSize(w, h)
        self.setMaximumSize(w, h)
        self.resetTransform()
        self.add_empty_photo()
        self.update()


    def add_empty_photo(self):
        self._photo = QGraphicsPixmapItem()
        self.scene.addItem(self._photo)

    def setPhoto(self, pixmap=None):
        if pixmap and not pixmap.isNull():
            # Get the size of the sceneRect
            targetSize = QSize(self.w, self.h)

            # Scale the pixmap to the new size with smooth transformation
            scaledPixmap = pixmap.scaled(targetSize,
                                         Qt.AspectRatioMode.IgnoreAspectRatio,
                                         Qt.TransformationMode.SmoothTransformation)

            self._photo.setPixmap(scaledPixmap)
            self.pixmap = scaledPixmap


class Canvas(QGraphicsView):
    endDrawing = pyqtSignal()

    def __init__(self, img_size):
        super().__init__()

        self.w, self.h = img_size

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setSceneRect(0, 0, self.w, self.h)
        self.setMinimumSize(self.w, self.h)
        self.setMaximumSize(self.w, self.h)

        self._photo = QGraphicsPixmapItem()
        self.scene.addItem(self._photo)
        self.pixmap = None  # a pixmap holder
        self.has_background = False  # no background to start with

        self.drawing_layer = QPixmap(self.w, self.h)
        self.drawing_layer.fill(Qt.GlobalColor.transparent)

        self.coloring_layer = QPixmap(self.w, self.h)
        self.coloring_layer.fill(Qt.GlobalColor.transparent)

        self.composed_drawing = QPixmap(self.w, self.h)
        self.composed_drawing.fill(Qt.GlobalColor.transparent)

        self.saved_drawing_layer = None  # To save the state of the drawing layer
        self.stored_pixmap = None  # To store the added pixmap

        self.active_layer = 0  # 0 is drawing, 1 is coloring
        self.tablet = False

        self.current_tool = 'brush'
        self.current_color = QColor(Qt.GlobalColor.black)
        self.brush_size = 10
        self.drawing = False
        self.last_point = None

        # bezier param
        self.create_bezier = False
        self.bez_nodes = []
        self.bez_handles = []
        self.bez_sym_handles = []
        self.bez_lines = []
        self.bez_sym_lines = []
        self.bez_curve_item = None
        self.bez_finalized = False  # To check if the curve has been finalized

        # custom paint cursor
        self.brush_cur = self.create_circle_cursor(10)
        self.temp_item = None

        self.setBackgroundBrush(QBrush(QColor(255, 255, 255)))
        self.setContentsMargins(0, 0, 0, 0)
        self.setViewportMargins(0, 0, 0, 0)

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.initHelpBox()
        self.hideHelpBox()

    # help box
    def initHelpBox(self):
        self.help_box_label = QLabel("Right click to stop!", self)
        self.help_box_label.setStyleSheet("""
            background-color: rgba(200, 200, 200, 200);
            border-radius: 10px;
            padding: 10px;
        """)
        self.help_box_label.setFont(QFont("Arial", 12))
        self.help_box_label.adjustSize()
        self.help_box_label.move(10, 10)  # Adjust position as needed

    def updateHelpText(self, text):
        self.help_box_label.setText(text)
        self.help_box_label.adjustSize()

    def hideHelpBox(self):
        """
        Hides the help box label.
        """
        self.help_box_label.setVisible(False)

    def showHelpBox(self):
        """
        Shows the help box label.
        """
        self.help_box_label.setVisible(True)

    def clean_scene(self):
        self.terminate_bezier()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        self._photo = QGraphicsPixmapItem()
        self.scene.addItem(self._photo)

        self.pixmap = None
        self.has_background = False

        # reset drawing zones
        self.drawing_layer = QPixmap(self.w, self.h)
        self.drawing_layer.fill(Qt.GlobalColor.transparent)

        self.coloring_layer = QPixmap(self.w, self.h)
        self.coloring_layer.fill(Qt.GlobalColor.transparent)

    def set_transparency(self, transparent):
        """ Set the transparency of the canvas. """
        if transparent:
            self.setWindowOpacity(0.5)  # Semi-transparent
        else:
            self.setWindowOpacity(1.0)  # Opaque

    def create_new_scene(self, w, h):
        self.w = w
        self.h = h
        self.scene.clear()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setSceneRect(0, 0, w, h)
        self.setMinimumSize(w, h)
        self.setMaximumSize(w, h)
        self.resetTransform()
        self.add_empty_photo()

        # reset drawing zone
        self.drawing_layer = QPixmap(self.w, self.h)
        self.drawing_layer.fill(Qt.GlobalColor.transparent)

        # Reset coloring layer
        self.coloring_layer = QPixmap(self.w, self.h)
        self.coloring_layer.fill(Qt.GlobalColor.transparent)

        # Reset composed drawing
        self.composed_drawing = QPixmap(self.w, self.h)
        self.composed_drawing.fill(Qt.GlobalColor.transparent)

        self.update()

    def scale_canvas(self, new_w, new_h):
        # Scale the background pixmap
        if self.pixmap:
            scaled_pixmap = self.pixmap.scaled(new_w, new_h, Qt.AspectRatioMode.KeepAspectRatio,
                                               Qt.TransformationMode.SmoothTransformation)
        if self.stored_pixmap:
            self.stored_pixmap = self.stored_pixmap.scaled(new_w, new_h, Qt.AspectRatioMode.KeepAspectRatio,
                                               Qt.TransformationMode.SmoothTransformation)

        # Scale the drawing layer
        scaled_drawing_layer = self.drawing_layer.scaled(new_w, new_h, Qt.AspectRatioMode.KeepAspectRatio,
                                                         Qt.TransformationMode.SmoothTransformation)

        # Scale the coloring layer
        scaled_coloring_layer = self.coloring_layer.scaled(new_w, new_h, Qt.AspectRatioMode.KeepAspectRatio,
                                                           Qt.TransformationMode.SmoothTransformation)

        self.create_new_scene(new_w, new_h)

        # Set the scaled background pixmap
        if self.pixmap:
            self.setPhoto(scaled_pixmap)

        # Update the drawing layer
        self.drawing_layer = QPixmap(new_w, new_h)
        self.drawing_layer.fill(Qt.GlobalColor.transparent)
        painter = QPainter(self.drawing_layer)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.drawPixmap(0, 0, scaled_drawing_layer)
        painter.end()

        # Update the coloring layer
        self.coloring_layer = QPixmap(new_w, new_h)
        self.coloring_layer.fill(Qt.GlobalColor.transparent)
        painter = QPainter(self.coloring_layer)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.drawPixmap(0, 0, scaled_coloring_layer)
        painter.end()

        self.update_drawing_coloring_layers()
        self.update_composite_pixmap()


    def add_empty_photo(self):
        self.pixmap = None
        self._photo = QGraphicsPixmapItem()
        self.scene.addItem(self._photo)

    def setPhoto(self, pixmap=None):
        if pixmap and not pixmap.isNull():
            self._photo.setPixmap(pixmap)
            self.pixmap = pixmap

    def set_image_in_drawing(self, pixmap):
                # Store the provided pixmap
        self.stored_pixmap = pixmap

        # Draw on the drawing layer
        painter = QPainter(self.drawing_layer)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw the stored pixmap on the drawing layer
        painter.drawPixmap(0, 0, self.stored_pixmap)

        painter.end()

        self.update_drawing_coloring_layers()
        self.update_composite_pixmap()

    def set_image_in_color_layer(self, pixmap):
        # Draw on the drawing layer
        painter = QPainter(self.coloring_layer)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw the stored pixmap on the drawing layer
        painter.drawPixmap(0, 0, pixmap)

        painter.end()

        self.update_drawing_coloring_layers()
        self.update_composite_pixmap()

    def remove_stored_pixmap(self):
        if self.stored_pixmap:
            # Clear the drawing layer by filling it with transparency
            self.drawing_layer.fill(Qt.GlobalColor.transparent)

            # Clear the stored pixmap
            self.stored_pixmap = None

            # Update the layers and composite pixmap
            self.update_drawing_coloring_layers()
            self.update_composite_pixmap()

    def change_to_brush_cursor(self):
        self.setCursor(self.brush_cur)

    def create_circle_cursor(self, diameter):
        # Create a QPixmap with a transparent background
        self.cursor_diameter = diameter

        scale_factor = self.transform().m11()
        # print(f'scale factor: {scale_factor}')
        scaledDiameter = int(diameter * scale_factor)  # Convert to integer

        pixmap = QPixmap(scaledDiameter, scaledDiameter)
        pixmap.fill(Qt.GlobalColor.transparent)

        # Create a QPainter to draw on the pixmap
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw a circle
        painter.setPen(QColor(Qt.GlobalColor.black))  # Black color, you can change as needed
        painter.drawEllipse(0, 0, scaledDiameter - 1, scaledDiameter - 1)

        # End painting
        painter.end()

        # Create a cursor from the pixmap
        return QCursor(pixmap)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.handle_press_event(event.pos(), event.button(), pressure=1.0)  # Default pressure for mouse
        elif event.button() == Qt.MouseButton.RightButton:
            self.terminate_bezier()
            self.endDrawing.emit()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton and self.drawing:
            self.handle_move_event(event.pos(), pressure=1.0)  # Default pressure for mouse
        elif event.buttons() == Qt.MouseButton.LeftButton and self.create_bezier:
            super().mouseMoveEvent(event)
            self.update_bezier(event.pos())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.drawing:
            self.endDrawing.emit()
            self.drawing = False
            self.temp_item = None  # Reset temporary item
        elif self.create_bezier:
            self.update_curve()

    def tabletEvent(self, event):
        pos = event.position().toPoint()  # Convert QPointF to QPoint
        pressure = event.pressure()  # Pressure is a float between 0.0 and 1.0
        self.tablet = True

        if event.type() == QEvent.Type.TabletPress:
            self.handle_press_event(pos, Qt.MouseButton.LeftButton, pressure)
        elif event.type() == QEvent.Type.TabletMove and self.drawing:
            self.handle_move_event(pos, pressure)
        elif event.type() == QEvent.Type.TabletRelease:
            if self.drawing:
                self.drawing = False
                self.tablet = False

        event.accept()

    def handle_press_event(self, pos, button, pressure):
        if button == Qt.MouseButton.LeftButton:
            if self.current_tool in ['brush', 'eraser', 'pencil', 'airbrush']:
                self.drawing = True
                self.create_bezier = False
                self.start_point = self.mapToScene(pos)
                self.last_point = pos
                self.current_path = QPainterPath(self.mapToScene(pos))
                if self.tablet:
                    self.set_pen_pressure(pressure)
                self.path_item = self.scene.addPath(self.current_path,
                                                    QPen(self.current_color, self.brush_size, Qt.PenStyle.SolidLine,
                                                         Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            elif self.current_tool in ['ellipse', 'rectangle']:
                self.start_point = self.mapToScene(pos)
                if self.current_tool == 'ellipse':
                    self.temp_item = QGraphicsEllipseItem(QRectF(self.start_point, self.start_point))
                elif self.current_tool == 'rectangle':
                    self.temp_item = QGraphicsRectItem(QRectF(self.start_point, self.start_point))

                if self.temp_item:
                    self.temp_item.setBrush(QBrush(self.current_color))
                    self.scene.addItem(self.temp_item)
            elif self.current_tool == 'bezier':
                self.create_bezier = True
                self.showHelpBox()
                self.add_bezier_control_point(pos)

    def handle_move_event(self, pos, pressure):
        end_point = self.mapToScene(pos)
        if self.current_tool in ['brush', 'eraser', 'pencil', 'airbrush']:
            self.draw_line(pos, pressure)
        elif self.current_tool in ['ellipse', 'rectangle'] and self.temp_item:
            self.update_temp_shape(end_point)

    def draw_line(self, end_point, pressure):
        if self.last_point is None:
            self.last_point = end_point  # Ensure this is a QPoint

        path = QPainterPath(self.mapToScene(self.last_point))
        path.lineTo(self.mapToScene(end_point))

        if self.tablet:
            self.set_pen_pressure(pressure)

        if self.active_layer == 0:
            # Draw on the drawing layer
            painter = QPainter(self.drawing_layer)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            pen = QPen(self.current_color, self.brush_size, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                       Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.drawPath(path)
            painter.end()
        elif self.active_layer == 1:
            # Draw on the coloring layer
            painter = QPainter(self.coloring_layer)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            pen = QPen(self.current_color, self.brush_size, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                       Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.drawPath(path)
            painter.end()

        # Composite black lines and color
        self.update_drawing_coloring_layers()

        # Composite the drawing layer over the background pixmap
        self.update_composite_pixmap()

        self.last_point = end_point

    def update_temp_shape(self, end_point):
        rect = QRectF(self.start_point, end_point).normalized()
        if self.current_tool == 'ellipse':
            self.temp_item.setRect(rect)
        elif self.current_tool == 'rectangle':
            self.temp_item.setRect(rect)

    def update_drawing_coloring_layers(self):
        # Create a new pixmap to hold the composite image
        composite_pixmap = QPixmap(self.w, self.h)
        composite_pixmap.fill(Qt.GlobalColor.transparent)

        # Draw the color pixmap
        painter = QPainter(composite_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.drawPixmap(0, 0, self.coloring_layer)

        # Draw the drawing layer using the mask
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Multiply)
        painter.drawPixmap(0, 0, self.drawing_layer)

        painter.end()
        self.composed_drawing = composite_pixmap

    def update_composite_pixmap(self):
        if not self.pixmap:
            self._photo.setPixmap(self.composed_drawing)
            return

        # Create a new pixmap to hold the composite image
        composite_pixmap = QPixmap(self.w, self.h)
        composite_pixmap.fill(Qt.GlobalColor.transparent)

        # Draw the background pixmap
        painter = QPainter(composite_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.drawPixmap(0, 0, self.pixmap)

        # Draw the drawing layer using the mask
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Multiply)
        painter.drawPixmap(0, 0, self.composed_drawing)

        painter.end()

        self._photo.setPixmap(composite_pixmap)

    def terminate_bezier(self):
        self.finalize_curve()
        self.reset_bezier()
        self.hideHelpBox()

    def reset_bezier(self):
        # bezier param
        self.bez_nodes = []
        self.bez_handles = []
        self.bez_sym_handles = []
        self.bez_lines = []
        self.bez_sym_lines = []
        self.bez_curve_item = None
        self.bez_finalized = False  # To check if the curve has been finalized

    def update_temp_line(self):
        print('update_temp_line')
        if len(self.bez_lines) > 0:
            cp = self.bez_nodes[-1].pos()
            hp = self.bez_handles[-1].pos()
            self.bez_lines[-1].setLine(cp.x(), cp.y(), hp.x(), hp.y())

    def update_temp_sym_line(self):
        print('update_sym_line')
        if len(self.bez_sym_lines) > 0:
            cp = self.bez_nodes[-1].pos()
            sp = self.bez_sym_handles[-1].pos()
            self.bez_sym_lines[-1].setLine(cp.x(), cp.y(), sp.x(), sp.y())

    def update_curve(self):
        print('update curve')
        if len(self.bez_nodes) < 2:
            print('not enough points')
            return

        path = QPainterPath()
        path.moveTo(self.bez_nodes[0].pos())

        for i in range(1, len(self.bez_nodes)):
            h1 = self.bez_handles[i - 1].pos()
            h2 = self.bez_sym_handles[i].pos()
            cp2 = self.bez_nodes[i].pos()
            path.cubicTo(h1, h2, cp2)

        if self.bez_curve_item:
            self.scene.removeItem(self.bez_curve_item)

        # Draw the updated bezier curve on the drawing layer
        painter = QPainter(self.drawing_layer)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor(Qt.GlobalColor.black), 2)
        painter.setPen(pen)
        painter.drawPath(path)
        painter.end()

        # Composite the drawing and coloring layers
        self.update_drawing_coloring_layers()

        # Update the composite pixmap
        self.update_composite_pixmap()

    def finalize_curve(self):
        print('finalize curve')
        if len(self.bez_nodes) < 2:
            return

        path = QPainterPath()
        path.moveTo(self.bez_nodes[0].pos())

        for i in range(1, len(self.bez_nodes)):
            h1 = self.bez_handles[i - 1].pos()
            h2 = self.bez_sym_handles[i].pos()
            cp2 = self.bez_nodes[i].pos()
            path.cubicTo(h1, h2, cp2)

        # Draw the bezier curve on the drawing layer
        painter = QPainter(self.drawing_layer)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor(Qt.GlobalColor.black), 2)
        painter.setPen(pen)
        painter.drawPath(path)
        painter.end()

        # Composite the drawing and coloring layers
        self.update_drawing_coloring_layers()

        # Update the composite pixmap
        self.update_composite_pixmap()

        # Hide and remove temporary bezier items from the scene
        for item in self.bez_nodes + self.bez_handles + self.bez_sym_handles + self.bez_lines + self.bez_sym_lines:
            self.scene.removeItem(item)

        self.bez_finalized = True
        self.reset_bezier()

    def add_bezier_control_point(self, pos):
        print('add control point')
        pos_scene = self.mapToScene(pos)

        control_point = QGraphicsEllipseItem(-5, -5, 10, 10)
        control_point.setPos(pos_scene)
        control_point.setBrush(QColor(Qt.GlobalColor.black))
        control_point.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, enabled=True)
        self.scene.addItem(control_point)
        self.bez_nodes.append(control_point)

        handle = QGraphicsEllipseItem(-3, -3, 6, 6)
        handle.setPos(pos_scene)
        handle.setBrush(QColor(100, 100, 100))
        handle.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, enabled=True)
        self.scene.addItem(handle)
        self.bez_handles.append(handle)

        sym_handle = QGraphicsEllipseItem(-3, -3, 6, 6)
        initial_sym_pos = QPointF(2 * control_point.pos().x() - pos_scene.x(),
                                  2 * control_point.pos().y() - pos_scene.y())
        sym_handle.setPos(initial_sym_pos)
        sym_handle.setBrush(QColor(Qt.GlobalColor.green))
        sym_handle.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, enabled=True)
        self.scene.addItem(sym_handle)
        self.bez_sym_handles.append(sym_handle)

        line = QGraphicsLineItem(control_point.pos().x(), control_point.pos().y(), handle.pos().x(),
                                 handle.pos().y())
        self.scene.addItem(line)
        self.bez_lines.append(line)

        sym_line = QGraphicsLineItem(control_point.pos().x(), control_point.pos().y(), sym_handle.pos().x(),
                                     sym_handle.pos().y())
        self.scene.addItem(sym_line)
        self.bez_sym_lines.append(sym_line)

    def update_bezier(self, pos):
        print('update bez')
        pos_scene = self.mapToScene(pos)
        if len(self.bez_handles) > 0:
            self.bez_handles[-1].setPos(pos_scene)
            self.update_temp_line()

            # Update symmetrical handle
            if len(self.bez_sym_handles) > 0:
                control_point_pos = self.bez_nodes[-1].pos()
                new_sym_pos = QPointF(2 * control_point_pos.x() - pos_scene.x(),
                                      2 * control_point_pos.y() - pos_scene.y())
                self.bez_sym_handles[-1].setPos(new_sym_pos)
                self.update_temp_sym_line()

    def smooth_and_finalize_path(self):
        if not self.current_path or self.current_path.isEmpty():
            return

        # Extract points from QPainterPath
        points = []
        num_elements = self.current_path.elementCount()
        for i in range(num_elements):
            elem = self.current_path.elementAt(i)
            points.append((elem.x, elem.y))

        if len(points) < 3:
            return  # Not enough points to smooth

        # Convert points to numpy array for processing
        points = np.array(points)
        x = points[:, 0]
        y = points[:, 1]

        # Fit spline to points
        try:
            spline = UnivariateSpline(x, y, s=len(points) * 1.5)  # Adjust smoothing factor based on your needs
            xs = np.linspace(x.min(), x.max(), 50)  # More points for a smoother curve
            ys = spline(xs)

            # Create a new QPainterPath with the smoothed curve
            smoothed_path = QPainterPath()
            smoothed_path.moveTo(xs[0], ys[0])
            for xi, yi in zip(xs[1:], ys[1:]):
                smoothed_path.lineTo(xi, yi)

            # Update the path item with the smoothed path
            if self.path_item:
                self.path_item.setPath(smoothed_path)

        except Exception as e:
            print("Error in smoothing:", e)

    def draw_ellipse(self, start_point, end_point):
        rect = QRectF(start_point, end_point)
        ellipse = QGraphicsEllipseItem(rect)
        ellipse.setBrush(QBrush(self.current_color))
        self.scene.addItem(ellipse)

    def draw_rectangle(self, start_point, end_point):
        rect = QRectF(start_point, end_point)
        rectangle = QGraphicsRectItem(rect)
        rectangle.setBrush(QBrush(self.current_color))
        self.scene.addItem(rectangle)

    def clear_drawing(self):
        for item in self.scene.items():
            if isinstance(item, QGraphicsPathItem) or \
                    isinstance(item, QGraphicsEllipseItem) or \
                    isinstance(item, QGraphicsRectItem):
                self.scene.removeItem(item)

    def set_tool(self, tool):
        self.current_tool = tool
        self.set_pen_color()

        if tool in ['brush', 'eraser', 'pencil', 'airbrush']:
            self.change_to_brush_cursor()
        else:
            print('cross cursor')
            self.setCursor(Qt.CursorShape.CrossCursor)

    def set_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.current_color = color

    def set_pen_color(self):
        if self.current_tool == 'brush':
            self.current_color = QColor(Qt.GlobalColor.black)
        elif self.current_tool == 'eraser':
            self.current_color = QColor(Qt.GlobalColor.white)
        elif self.current_tool == 'pencil':
            self.current_color = QColor(45, 45, 45)
        elif self.current_tool == 'airbrush':
            self.current_color = QColor(120, 45, 45)
        else:
            self.current_color = Qt.GlobalColor.black

        print(f'tool {self.current_tool} set')

    def set_pen_pressure(self, pressure):
        if self.current_tool == 'brush':
            pen_size = max(1, min(int(pressure * 50), 50))  # Dynamic pen size based on pressure
        elif self.current_tool == 'eraser':
            pen_size = max(1, min(int(pressure * 50), 50))  # Dynamic pen size based on pressure
        elif self.current_tool == 'pencil':
            pen_size = max(1, min(int(pressure * 12), 12))
        elif self.current_tool == 'airbrush':
            pen_size = max(10, min(int(pressure * 100), 100))
        else:
            pen_size = max(1, min(int(pressure * 50), 50))

        self.brush_size = pen_size

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        self.brush_size += delta / 120  # Adjust this factor if needed
        self.brush_size = max(1, min(self.brush_size, 50))  # Limit brush size

        self.brush_cur = self.create_circle_cursor(self.brush_size)
        self.change_to_brush_cursor()
