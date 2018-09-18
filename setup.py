from setuptools import setup

setup(name='face-processor',
      version='0.1',
      description='Aligns faces in videos and images',
      packages=['face-processor'],
      package_dir={'face-processor': 'face-processor'},
      package_data={'syncnet': ['data/*.npy']},
      install_requires=[
          'face_alignment',
          'scikit-video',
          'opencv-python',
          'scikit-image'
      ],
      zip_safe=False)

