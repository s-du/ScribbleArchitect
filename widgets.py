# standard libraries
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
        self.pixmap = None

        self.drawing_layer = QPixmap(self.w, self.h)
        self.drawing_layer.fill(Qt.GlobalColor.transparent)

        self.current_tool = 'brush'
        self.current_color = QColor(Qt.GlobalColor.black)
        self.brush_size = 10
        self.drawing = False
        self.last_point = None

        self.has_background = False

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

        # reset drawing zone
        self.drawing_layer = QPixmap(self.w, self.h)
        self.drawing_layer.fill(Qt.GlobalColor.transparent)


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

        self.update()

    def add_empty_photo(self):
        self.pixmap = None
        self._photo = QGraphicsPixmapItem()
        self.scene.addItem(self._photo)

    def setPhoto(self, pixmap=None):
        if pixmap and not pixmap.isNull():
            self._photo.setPixmap(pixmap)
            self.pixmap = pixmap

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
    # bezier
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
        if len(self.bez_lines) > 0:
            cp = self.bez_nodes[-1].pos()
            hp = self.bez_handles[-1].pos()
            self.bez_lines[-1].setLine(cp.x(), cp.y(), hp.x(), hp.y())

    def update_temp_sym_line(self):
        if len(self.bez_sym_lines) > 0:
            cp = self.bez_nodes[-1].pos()
            sp = self.bez_sym_handles[-1].pos()
            self.bez_sym_lines[-1].setLine(cp.x(), cp.y(), sp.x(), sp.y())

    def update_curve(self):
        if len(self.bez_nodes) < 2:
            return

        path = QPainterPath()
        path.moveTo(self.bez_nodes[0].pos())

        for i in range(1, len(self.bez_nodes)):
            cp1 = self.bez_nodes[i - 1].pos()
            h1 = self.bez_handles[i - 1].pos()
            h2 = self.bez_sym_handles[i].pos()
            cp2 = self.bez_nodes[i].pos()
            path.cubicTo(h1, h2, cp2)

        if self.bez_curve_item:
            self.scene.removeItem(self.bez_curve_item)
        self.bez_curve_item = self.scene.addPath(path, QPen(QColor(Qt.GlobalColor.black), 2))

    def finalize_curve(self):
        self.bez_finalized = True
        for item in self.bez_nodes + self.bez_handles + self.bez_sym_handles + self.bez_lines + self.bez_sym_lines:
            item.setVisible(False)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.current_tool in ['brush', 'eraser','pencil']:
                self.drawing = True
                self.create_bezier = False
                self.start_point = self.mapToScene(event.pos())
                self.last_point = event.pos()

            elif self.current_tool in ['ellipse', 'rectangle']:
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
                pos = self.mapToScene(event.pos())

                control_point = QGraphicsEllipseItem(-5, -5, 10, 10)
                control_point.setPos(pos)
                control_point.setBrush(QColor(Qt.GlobalColor.black))
                control_point.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, enabled=True)
                self.scene.addItem(control_point)
                self.bez_nodes.append(control_point)

                handle = QGraphicsEllipseItem(-3, -3, 6, 6)
                handle.setPos(pos)
                handle.setBrush(QColor(100,100,100))
                handle.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, enabled=True)
                self.scene.addItem(handle)
                self.bez_handles.append(handle)

                sym_handle = QGraphicsEllipseItem(-3, -3, 6, 6)
                initial_sym_pos = QPointF(2 * control_point.pos().x() - pos.x(), 2 * control_point.pos().y() - pos.y())
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

        elif event.button() == Qt.MouseButton.RightButton:
            self.terminate_bezier()
            self.endDrawing.emit()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton and self.drawing:
            end_point = self.mapToScene(event.pos())
            if self.current_tool in ['brush', 'eraser','pencil']:
                if self.current_tool in ['brush', 'pencil']:
                    self.draw_line(event.pos())
                elif self.current_tool == 'eraser':
                    self.erase_line(event.pos())
            elif self.current_tool in ['ellipse', 'rectangle'] and self.temp_item:
                self.update_temp_shape(end_point)

        elif event.buttons() == Qt.MouseButton.LeftButton and self.create_bezier:
            super().mouseMoveEvent(event)
            pos = self.mapToScene(event.pos())
            if len(self.bez_handles) > 0:
                self.bez_handles[-1].setPos(pos)
                self.update_temp_line()

                # Update symmetrical handle
                if len(self.bez_sym_handles) > 0:
                    control_point_pos = self.bez_nodes[-1].pos()
                    new_sym_pos = QPointF(2 * control_point_pos.x() - pos.x(),
                                          2 * control_point_pos.y() - pos.y())
                    self.bez_sym_handles[-1].setPos(new_sym_pos)
                    self.update_temp_sym_line()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.drawing:
            self.endDrawing.emit()
            self.drawing = False
            self.temp_item = None  # Reset temporary item

        elif self.create_bezier:
            self.update_curve()

    def update_temp_shape(self, end_point):
        rect = QRectF(self.start_point, end_point).normalized()
        if self.current_tool == 'ellipse':
            self.temp_item.setRect(rect)
        elif self.current_tool == 'rectangle':
            self.temp_item.setRect(rect)

    def erase_line(self, end_point):
        if self.last_point is None:
            self.last_point = end_point  # Ensure this is a QPoint

        eraser_path = QPainterPath(self.mapToScene(self.last_point))
        eraser_path.lineTo(self.mapToScene(end_point))

        # Draw on the drawing layer
        painter = QPainter(self.drawing_layer)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(Qt.GlobalColor.white, self.brush_size, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                   Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.drawPath(eraser_path)
        painter.end()

        # Composite the drawing layer over the background pixmap
        self.update_composite_pixmap()

        self.last_point = end_point

    def draw_line(self, end_point):
        if self.last_point is None:
            self.last_point = end_point  # Ensure this is a QPoint

        path = QPainterPath(self.mapToScene(self.last_point))
        path.lineTo(self.mapToScene(end_point))

        # Draw on the drawing layer
        painter = QPainter(self.drawing_layer)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(Qt.GlobalColor.black, self.brush_size, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                   Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.drawPath(path)
        painter.end()

        # Composite the drawing layer over the background pixmap
        self.update_composite_pixmap()

        self.last_point = end_point

    def update_composite_pixmap(self):
        if not self.pixmap:
            self._photo.setPixmap(self.drawing_layer)
            return

        # Create a new pixmap to hold the composite image
        composite_pixmap = QPixmap(self.w, self.h)
        composite_pixmap.fill(Qt.GlobalColor.transparent)

        # Draw the background pixmap
        painter = QPainter(composite_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.drawPixmap(0, 0, self.pixmap)

        # Set the composition mode to SourceOver
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Multiply)

        # Draw the drawing layer using the mask
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Multiply)
        painter.drawPixmap(0, 0, self.drawing_layer)

        painter.end()

        self._photo.setPixmap(composite_pixmap)

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

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        self.brush_size += delta / 120  # Adjust this factor if needed
        self.brush_size = max(1, min(self.brush_size, 50))  # Limit brush size

        self.brush_cur = self.create_circle_cursor(self.brush_size)
        self.change_to_brush_cursor()

    def tabletEvent(self, event):
        pos = event.position().toPoint()  # Convert QPointF to QPoint
        pressure = event.pressure()  # Pressure is a float between 0.0 and 1.0

        pen_size = self.set_pen_pressure(pressure)

        if event.type() == QEvent.Type.TabletPress:
            self.drawing = True
            self.current_path = QPainterPath()
            self.current_path.moveTo(self.mapToScene(pos))
            self.last_pen_size = pen_size
            self.create_new_segment(pen_size)

        elif event.type() == QEvent.Type.TabletMove and self.drawing:
            if self.last_pen_size != pen_size:
                # Finish the current segment and start a new one
                self.create_new_segment(pen_size)

            # Continue the path with the new or existing pen size
            self.current_path.lineTo(self.mapToScene(pos))
            self.path_item.setPath(self.current_path)

        elif event.type() == QEvent.Type.TabletRelease:
            if self.drawing:
                self.drawing = False
                self.endDrawing.emit()
                # self.smooth_and_finalize_path()

        event.accept()

    def create_new_segment(self, pen_size):
        if self.current_path:
            # Start a new path from the last point
            new_path = QPainterPath()
            new_path.moveTo(self.current_path.currentPosition())
            self.current_path = new_path
            self.path_item = self.scene.addPath(self.current_path,
                                                QPen(self.current_color, pen_size, Qt.PenStyle.SolidLine,
                                                     Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            self.last_pen_size = pen_size

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

    def set_tool(self, tool):
        self.current_tool = tool
        self.set_pen_color()

        if tool in ['brush','eraser','pencil']:
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
            self.current_color = QColor(45,45,45)
        else:
            self.current_color = Qt.GlobalColor.black

    def set_pen_pressure(self, pressure):
        if self.current_tool == 'brush':
            pen_size = max(1, min(int(pressure * 50), 50))  # Dynamic pen size based on pressure
        elif self.current_tool == 'eraser':
            pen_size = max(1, min(int(pressure * 50), 50))  # Dynamic pen size based on pressure
        elif self.current_tool == 'pencil':
            pen_size = max(1, min(int(pressure * 15), 15))
        else:
            pen_size = max(1, min(int(pressure * 50), 50))

        return pen_size
