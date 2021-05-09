import serial
import time
from time import sleep
from onvif import ONVIFCamera
import zeep

def zeep_pythonvalue(self, xmlvalue):
    return xmlvalue


pan, pan_speed = 0, 1
tilt, tilt_speed = 0, 1
zoom, zoom_speed = 0, 1
mycam = ONVIFCamera('192.168.1.88', 8080, 'admin', 'admin',
                    wsdl_dir=r"C:\Users\Cristhiam\AppData\Local\Programs\Python\Python38\Lib\site-packages\onvif_zeep-0.2.12-py3.8.egg\Lib\site-packages\wsdl")
# Create media service object
media = mycam.create_media_service()
# Create ptz service object
ptz = mycam.create_ptz_service()
# Get target profile
zeep.xsd.simple.AnySimpleType.pythonvalue = zeep_pythonvalue
media_profile = media.GetProfiles()[1]


def absolute_move(zoom):
    # Get PTZ configuration options for getting absolute move range
    request = ptz.create_type('GetConfigurationOptions')
    request.ConfigurationToken = media_profile.PTZConfiguration.token
    # ptz_configuration_options = ptz.GetConfigurationOptions(request)
    request = ptz.create_type('AbsoluteMove')
    request.ProfileToken = media_profile.token
    # ptz.Stop({'ProfileToken': media_profile.token})
    if request.Position is None:
        request.Position = ptz.GetStatus({'ProfileToken': media_profile.token}).Position
    if request.Speed is None:
        request.Speed = ptz.GetStatus({'ProfileToken': media_profile.token}).Position
    request.Position.PanTilt.x = pan
    request.Speed.PanTilt.x = pan_speed
    request.Position.PanTilt.y = tilt
    request.Speed.PanTilt.y = tilt_speed
    request.Position.Zoom = zoom
    request.Speed.Zoom = zoom_speed
    ptz.AbsoluteMove(request)
    sleep(0.1)
    zoom = 0
    request.Position.Zoom = zoom
    ptz.AbsoluteMove(request)


def control_zoom(width, height, bbox):
    '''
    :param width: width of the image
    :param height: height of the image
    :param bbox: bounding box coordinates [left_x, top_y, right_x, down_y]
    :return:
    '''
    left_x, top_y, right_x, down_y = bbox[0], bbox[1], bbox[2], bbox[3]
    box_height = right_x - left_x
    box_width = down_y - top_y
    bbox_area = box_height * box_width
    image_area = width * height
    print(bbox_area)
    print(image_area)
    if bbox_area > (0.1 * image_area):
        absolute_move(-1)
        print("aleja")
    elif bbox_area < (0.03 * image_area):
        absolute_move(1)
        print("acerca")


arduinoPort = 'COM4'
arduinoBaudRate = 9600
ser = serial.Serial(arduinoPort, arduinoBaudRate, timeout=1)
# wait so that arduino can reset
time.sleep(2)
angle = 2


def move_camera(width, height, bbox, threshold=0.1):
    '''
    :param width: width of the image
    :param height: height of the image
    :param bbox: bounding box coordinates [left_x, top_y, right_x, down_y]
    :param threshold: limite a partir del cual es necesario mover la camara. (valor entre 0 y 1)
    :return:
    '''
    pan_angle = 0
    tilt_angle = 0
    left_x, top_y, right_x, down_y = bbox
    if (top_y < int(threshold * height)) and (down_y > (height - (threshold * height))):
        pass
    elif top_y < int(threshold * height):
        tilt_angle = angle
    elif down_y > (height - (threshold * height)):
        tilt_angle = -angle
    if (left_x < int(threshold * width)) and (right_x > (width - (threshold * width))):
        pass
    elif left_x < int(threshold * width):
        pan_angle = angle
    elif right_x > (width - (threshold * width)):
        pan_angle = -angle

    data = "P{0:d}T{1:d}".format(pan_angle, tilt_angle)
    ser.write(data.encode())
