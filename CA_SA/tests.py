from PyQt5.QtWidgets import QApplication, QOpenGLWidget
from PyQt5.QtGui import QImage, QOpenGLShader, QOpenGLShaderProgram, QOpenGLTexture
from PyQt5.QtCore import Qt, QPoint


class ImageRenderer(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None
        self.texture = None

    def initializeGL(self):
        self.initializeOpenGLFunctions()
        self.glClearColor(0.0, 0.0, 0.0, 1.0)

    def resizeGL(self, width, height):
        self.glViewport(0, 0, width, height)

    def paintGL(self):
        self.glClear(self.GL_COLOR_BUFFER_BIT)
        if self.image is not None:
            self.texture.bind()
            self.drawTexture(self.rect(), self.texture.textureId(), self.texture.target())

    def setImage(self, image):
        if image is None:
            return

        self.makeCurrent()

        # 创建纹理对象
        self.texture = QOpenGLTexture(image)
        self.texture.setMinificationFilter(QOpenGLTexture.Linear)
        self.texture.setMagnificationFilter(QOpenGLTexture.Linear)
        self.texture.setWrapMode(QOpenGLTexture.ClampToEdge)

        self.doneCurrent()


if __name__ == '__main__':
    # 创建应用程序和窗口
    app = QApplication([])
    window = ImageRenderer()

    # 加载图像
    image_path = 'images.jpg'  # 替换为您的图像路径
    image = QImage(image_path)

    # 设置图像
    window.setImage(image)

    # 显示窗口
    window.show()
    app.exec_()
