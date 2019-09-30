import subprocess, pathlib


subprocess.call(['mscore', '-S', str(pathlib.Path.home()) + '/Documents/MuseScore2/Styles/custom_style.mss', 'temp.musicxml', '-o', 'temp.mscx'])
subprocess.call(['mscore', 'temp.mscx', '-o', 'temp.png'])
subprocess.call(['mscore', 'temp.mscx', '-o', 'temp.svg'])