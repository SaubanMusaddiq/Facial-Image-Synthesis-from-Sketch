# Facial-Image-Synthesis-from-Sketch

Generates photo realistic images from pencil sketches using Fixed-point GAN (Generative Adversarial
Network), Cycle GAN and StarGAN.
Fix point P GAN generates facial images that have dramatically reduced artifacts compared to other generative models.

<table align='center'>
<tr>
<td><img src = '/images/cycle_gan_10k.png'>
</tr>
<tr align='center'>
<td>Cycle Gan Generations. Contains 2 generators: G1 for sketch to facial image and G2 for facial Image to sketch. Row 1: CelebA Skethces, Row2: G1 output, Row3: CelebA images, Row4: G2 output</td>
</tr>
</table>

References :

GAN Paper : https://arxiv.org/abs/1406.2661
</br>
DCGAN Paper : https://arxiv.org/abs/1411.1784
</br>
Star Gan: https://arxiv.org/abs/1711.09020
</br>
Fixed Point Gan: https://arxiv.org/abs/1908.06965
</br>

https://github.com/eriklindernoren/PyTorch-GAN
</br>
https://github.com/yunjey/stargan
</br>
https://github.com/mahfuzmohammad/Fixed-Point-GAN
</br>
