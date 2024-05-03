# standard libraries
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

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

        self.current_tool = 'brush'
        self.current_color = QColor(Qt.GlobalColor.black)
        self.brush_size = 10
        self.drawing = False
        self.last_point = None

        # custom paint cursor
        self.brush_cur = self.create_circle_cursor(10)

        self.temp_item = None

        self.setBackgroundBrush(QBrush(QColor(255, 255, 255)))
        self.setContentsMargins(0, 0, 0, 0)
        self.setViewportMargins(0, 0, 0, 0)

        self.setRenderHint(QPainter.RenderHint.Antialiasing)

    def clean_scene(self):
        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        self._photo = QGraphicsPixmapItem()
        self.scene.addItem(self._photo)

    def set_transparency(self, transparent):
        """ Set the transparency of the canvas. """
        if transparent:
            self.setWindowOpacity(0.5)  # Semi-transparent
        else:
            self.setWindowOpacity(1.0)  # Opaque

    def create_new_scene(self, w, h):
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
            self._photo.setPixmap(pixmap)

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
            self.drawing = True
            self.start_point = self.mapToScene(event.pos())
            self.last_point = event.pos()

            if self.current_tool in ['ellipse', 'rectangle']:
                if self.current_tool == 'ellipse':
                    self.temp_item = QGraphicsEllipseItem(QRectF(self.start_point, self.start_point))
                elif self.current_tool == 'rectangle':
                    self.temp_item = QGraphicsRectItem(QRectF(self.start_point, self.start_point))

                if self.temp_item:
                    self.temp_item.setBrush(QBrush(self.current_color))
                    self.scene.addItem(self.temp_item)

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton and self.drawing:
            end_point = self.mapToScene(event.pos())
            if self.current_tool in ['brush', 'eraser']:
                if self.current_tool == 'brush':
                    self.draw_line(event.pos())
                elif self.current_tool == 'eraser':
                    self.erase_line(event.pos())
            elif self.current_tool in ['ellipse', 'rectangle'] and self.temp_item:
                self.update_temp_shape(end_point)

    def mouseReleaseEvent(self, event):
        self.endDrawing.emit()
        if event.button() == Qt.MouseButton.LeftButton and self.drawing:
            self.drawing = False
            self.temp_item = None  # Reset temporary item

    def update_temp_shape(self, end_point):
        rect = QRectF(self.start_point, end_point).normalized()
        if self.current_tool == 'ellipse':
            self.temp_item.setRect(rect)
        elif self.current_tool == 'rectangle':
            self.temp_item.setRect(rect)

    def draw_line(self, end_point):
        path = QPainterPath(self.mapToScene(self.last_point))
        path.lineTo(self.mapToScene(end_point))
        pen = QPen(self.current_color, self.brush_size, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        self.scene.addPath(path, pen)
        self.last_point = end_point

    def erase_line(self, end_point):
        eraser_path = QPainterPath(self.mapToScene(self.last_point))
        eraser_path.lineTo(self.mapToScene(end_point))
        eraser = QPen(Qt.GlobalColor.white, self.brush_size, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        self.scene.addPath(eraser_path, eraser)
        self.last_point = end_point

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

    def set_tool(self, tool):
        self.current_tool = tool
        if tool == 'brush' or 'eraser':
            self.change_to_brush_cursor()
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def set_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.current_color = color
