from setuptools import setup

setup(name='face_processor',
      version='0.1',
      description='Aligns faces in videos and images',
      packages=['face_processor'],
      package_dir={'face_processor': 'face_processor'},
      package_data={'face_processor': ['data/*.npy']},
      install_requires=[
          'face-alignment',
          'scikit-video',
          'opencv-python',
          'scikit-image', 
          'progressbar'
      ],
      zip_safe=False)

