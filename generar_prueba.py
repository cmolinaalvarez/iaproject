from PIL import Image, ImageDraw, ImageFont

# Crear una imagen en blanco (fondo de papel)
imagen = Image.new('RGB', (800, 300), color=(255, 255, 255))
d = ImageDraw.Draw(imagen)

# Texto de prueba (Pangrama)
texto = "El veloz murcielago hindu comia feliz cardillo y kiwi."

# Dibujar el texto en la imagen (usando la fuente por defecto del sistema)
d.text((50, 100), texto, fill=(20, 20, 80))

# Guardar la imagen de prueba
imagen.save('manuscrito_prueba.png')
print("¡Imagen 'manuscrito_prueba.png' generada con éxito en tu carpeta!")