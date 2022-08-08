
from paraview.simple import *

def ResetSession():
    pxm = servermanager.ProxyManager()
    pxm.UnRegisterProxies()
    del pxm
    Disconnect()
    Connect()

def saveimg(filename, colorTransferFunction):
	ResetSession()
	reader_name = filename+'.pvd'
	reader = PVDReader(FileName=[reader_name])
	Show()
	ResetCamera()
	camera=GetActiveCamera()
	camera.SetFocalPoint(0,0,0.5)
	camera.SetPosition(0,0,2.1)
	camera.SetViewUp(0,1,0)

	camera.Azimuth(-90)
	Render()

	myview = GetActiveView()
	myview.Background = [1,1,1]

	colorMap = GetColorTransferFunction(colorTransferFunction)

	scalarBar = GetScalarBar(colorMap, myview)
	scalarBar.TitleColor = [0.0,0.0,0.0]
	scalarBar.Title = [filename]
	scalarBar.LabelColor = [0.0,0.0,0.0]
	scalarBar.Visibility = True
	#scalarBar.Orientation = 'Horizantal'
	scalarBar.RangeLabelFormat = ['%-#6.1f']

	Render()
	ss = filename+'.png'
	SaveScreenshot(ss, myview,
		ImageResolution=[1920, 1080],
		TransparentBackground=0)
	print("Image", filename, "saved..")
	
name = ['p_r', 'p_i', 'p_abs']
ctf = ['f_66-0', 'f_66-1', 'f_85']
#name = ['p_i']
for i in range(len(name)):
	saveimg(name[i], ctf[i])
