from ultralytics import YOLO
import PySpin
import matplotlib.pyplot as plt
import keyboard
import numpy
import cv2
import argparse

continue_recording = True

def handle_close(evt):
    """
    This function will close the GUI when close event happens.

    :param evt: Event that occurs when the figure closes.
    :type evt: Event
    """

    global continue_recording
    continue_recording = False


def set_thermal_properties(nodemap):
    """
    This function sets the thermal properties of the AX5 to be able to get temperature readings.

    :param nodemap: Device nodemap.
    :type nodemap: INodeMap
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    # Set the pixel format to 14 bit
    node_pixel_format = PySpin.CEnumerationPtr(nodemap.GetNode('PixelFormat'))
    if not PySpin.IsAvailable(node_pixel_format) or not PySpin.IsWritable(node_pixel_format):
        print('Unable to set pixel format.. Aborting...')
        return False
    node_pixel_format_mono14 = PySpin.CEnumEntryPtr(node_pixel_format.GetEntryByName('Mono14'))
    if not PySpin.IsAvailable(node_pixel_format_mono14) or not PySpin.IsReadable(node_pixel_format_mono14):
        print('Unable to set pixel format.. Aborting...')
        return False
    pixel_format_mono14 = node_pixel_format_mono14.GetValue()
    node_pixel_format.SetIntValue(pixel_format_mono14)

    # Set the temperature resolution to high
    node_temp_linear = PySpin.CEnumerationPtr(nodemap.GetNode('TemperatureLinearResolution'))
    if not PySpin.IsAvailable(node_temp_linear) or not PySpin.IsWritable(node_temp_linear):
        print('Unable to set temperature resolution.. Aborting...')
        return False
    node_temp_linear_high = PySpin.CEnumEntryPtr(node_temp_linear.GetEntryByName('High'))
    if not PySpin.IsAvailable(node_temp_linear_high) or not PySpin.IsReadable(node_temp_linear_high):
        print('Unable to set temperature resolution.. Aborting...')
        return False
    linear_high = node_temp_linear_high.GetValue()
    node_temp_linear.SetIntValue(linear_high)

    # Set the CMOS bit depth to 14
    node_bit_depth = PySpin.CEnumerationPtr(nodemap.GetNode('CMOSBitDepth'))
    if not PySpin.IsAvailable(node_bit_depth) or not PySpin.IsWritable(node_bit_depth):
        print('Unable to set CMOS bit depth.. Aborting...')
        return False
    node_bit_depth_14bit = PySpin.CEnumEntryPtr(node_bit_depth.GetEntryByName('bit14bit'))
    if not PySpin.IsAvailable(node_bit_depth_14bit) or not PySpin.IsReadable(node_bit_depth_14bit):
        print('Unable to set CMOS bit depth.. Aborting...')
        return False
    bit_depth = node_bit_depth_14bit.GetValue()
    node_bit_depth.SetIntValue(bit_depth)

    # Turn on temperature linear mode
    node_temp_linear = PySpin.CEnumerationPtr(nodemap.GetNode('TemperatureLinearMode'))
    if not PySpin.IsAvailable(node_temp_linear) or not PySpin.IsWritable(node_temp_linear):
        print('Unable to set temperature linear mode.. Aborting...')
        return False
    node_temp_linear_on = PySpin.CEnumEntryPtr(node_temp_linear.GetEntryByName('On'))
    if not PySpin.IsAvailable(node_temp_linear_on) or not PySpin.IsReadable(node_temp_linear_on):
        print('Unable to set temperature linear mode.. Aborting...')
        return False
    node_on = node_temp_linear_on.GetValue()
    node_temp_linear.SetIntValue(node_on)

    return True


def acquire_and_display_images(cam, nodemap, nodemap_tldevice):
    """
    This function continuously acquires images from a device and display them in a GUI.

    :param cam: Camera to acquire images from.
    :param nodemap: Device nodemap.
    :param nodemap_tldevice: Transport layer device nodemap.
    :type cam: CameraPtr
    :type nodemap: INodeMap
    :type nodemap_tldevice: INodeMap
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    global continue_recording

    if not set_thermal_properties(nodemap):
        return False

    sNodemap = cam.GetTLStreamNodeMap()

    # Change buffer handling mode to NewestOnly
    node_bufferhandling_mode = PySpin.CEnumerationPtr(sNodemap.GetNode('StreamBufferHandlingMode'))
    if not PySpin.IsAvailable(node_bufferhandling_mode) or not PySpin.IsWritable(node_bufferhandling_mode):
        print('Unable to set stream buffer handling mode.. Aborting...')
        return False
    # Retrieve entry node from enumeration node
    node_newestonly = node_bufferhandling_mode.GetEntryByName('NewestOnly')
    if not PySpin.IsAvailable(node_newestonly) or not PySpin.IsReadable(node_newestonly):
        print('Unable to set stream buffer handling mode.. Aborting...')
        return False
    # Retrieve integer value from entry node
    node_newestonly_mode = node_newestonly.GetValue()
    # Set integer value from entry node as new value of enumeration node
    node_bufferhandling_mode.SetIntValue(node_newestonly_mode)

    print('*** IMAGE ACQUISITION ***\n')
    try:
        node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
        if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
            print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
            return False

        # Retrieve entry node from enumeration node
        node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
        if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(
                node_acquisition_mode_continuous):
            print('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
            return False
        # Retrieve integer value from entry node
        acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()
        # Set integer value from entry node as new value of enumeration node
        node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

        print('Acquisition mode set to continuous...')

        #  Begin acquiring images
        #
        #  *** NOTES ***
        #  What happens when the camera begins acquiring images depends on the
        #  acquisition mode. Single frame captures only a single image, multi
        #  frame catures a set number of images, and continuous captures a
        #  continuous stream of images.
        #
        #  *** LATER ***
        #  Image acquisition must be ended when no more images are needed.
        cam.BeginAcquisition()

        print('Acquiring images...')

        #  Retrieve device serial number for filename
        #
        #  *** NOTES ***
        #  The device serial number is retrieved in order to keep cameras from
        #  overwriting one another. Grabbing image IDs could also accomplish
        #  this.
        device_serial_number = ''
        node_device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
        if PySpin.IsAvailable(node_device_serial_number) and PySpin.IsReadable(node_device_serial_number):
            device_serial_number = node_device_serial_number.GetValue()
            print('Device serial number retrieved as %s...' % device_serial_number)

        # Close program
        print('Press enter to close the program..')

        # Figure(1) is default so you can omit this line. Figure(0) will create a new window every time program hits this line
        fig = plt.figure(1)

        # Close the GUI when close event happens
        fig.canvas.mpl_connect('close_event', handle_close)

        # Retrieve and display images
        while(continue_recording):
            try:

                #  Retrieve next received image
                #
                #  *** NOTES ***
                #  Capturing an image houses images on the camera buffer. Trying
                #  to capture an image that does not exist will hang the camera.
                #
                #  *** LATER ***
                #  Once an image from the buffer is saved and/or no longer
                #  needed, the image must be released in order to keep the
                #  buffer from filling up.

                image_result = cam.GetNextImage()

                #  Ensure image completion
                if image_result.IsIncomplete():
                    print('Image incomplete with image status %d ...' % image_result.GetImageStatus())

                else:
                    # Getting the image data as a numpy array
                    image_data = image_result.GetNDArray()
                    image_data_celsius = image_data * 0.04 - 273.15

                    max_celsius = numpy.max(image_data_celsius)
                    print("Maximum temperature in frame: ", max_celsius)

                    # Draws an image on the current figure
                    plt.imshow(image_data_celsius, cmap='inferno')
                    plt.colorbar(format='%.2f')

                    # Interval in plt.pause(interval) determines how fast the images are displayed in a GUI
                    # Interval is in seconds.
                    plt.pause(0.001)

                    # Clear current reference of a figure. This will improve display speed significantly
                    plt.clf()

                    # If user presses enter, close the program
                    if keyboard.is_pressed('ENTER'):
                        print('Program is closing...')

                        # Close figure
                        plt.close('all')
                        input('Done! Press Enter to exit...')
                        continue_recording = False

                #  Release image
                #
                #  *** NOTES ***
                #  Images retrieved directly from the camera (i.e. non-converted
                #  images) need to be released in order to keep from filling the
                #  buffer.
                image_result.Release()

            except PySpin.SpinnakerException as ex:
                print('Error: %s' % ex)
                return False

        #  End acquisition
        #
        #  *** NOTES ***
        #  Ending acquisition appropriately helps ensure that devices clean up
        #  properly and do not need to be power-cycled to maintain integrity.
        cam.EndAcquisition()

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return True

def acquire_and_infer_yolo(cam, nodemap, nodemap_tldevice, model_path=None, conf=0.25, imgsz=640, device='cpu'):
    """
    Acquire thermal frames from the camera and run a pretrained YOLO model for inference.

    :param cam: Camera to acquire images from.
    :param nodemap: Device nodemap.
    :param nodemap_tldevice: Transport layer device nodemap.
    :param model_path: Path to a pretrained ultralytics YOLO model (optional). If None, the default 'yolov8n.pt' will be attempted if ultralytics is installed.
    :param conf: Confidence threshold for detections.
    :param imgsz: Inference image size (square). Frames will be resized to this for the model.
    :param device: Device for inference ('cpu' or 'cuda').
    :return: True if successful, False otherwise.
    """
    global continue_recording

    if not set_thermal_properties(nodemap):
        return False

    model_name = model_path or 'yolov11n.pt'
    try:
        model = YOLO(model_name)
        # set device via model.predict arguments or model.to(device) depending on ultralytics version
    except Exception as ex:
        print(f'Failed to load YOLO model {model_name}: {ex}')
        return False

    sNodemap = cam.GetTLStreamNodeMap()

    # Change buffer handling mode to NewestOnly
    node_bufferhandling_mode = PySpin.CEnumerationPtr(sNodemap.GetNode('StreamBufferHandlingMode'))
    if not PySpin.IsAvailable(node_bufferhandling_mode) or not PySpin.IsWritable(node_bufferhandling_mode):
        print('Unable to set stream buffer handling mode.. Aborting...')
        return False
    # Retrieve entry node from enumeration node
    node_newestonly = node_bufferhandling_mode.GetEntryByName('NewestOnly')
    if not PySpin.IsAvailable(node_newestonly) or not PySpin.IsReadable(node_newestonly):
        print('Unable to set stream buffer handling mode.. Aborting...')
        return False
    # Retrieve integer value from entry node
    node_newestonly_mode = node_newestonly.GetValue()
    # Set integer value from entry node as new value of enumeration node
    node_bufferhandling_mode.SetIntValue(node_newestonly_mode)

    try:
        node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
        if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
            print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
            return False

        # Retrieve entry node from enumeration node
        node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
        if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(
                node_acquisition_mode_continuous):
            print('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
            return False
        # Retrieve integer value from entry node
        acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()
        # Set integer value from entry node as new value of enumeration node
        node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

        print('Acquisition mode set to continuous...')

        #  Begin acquiring images
        #
        #  *** NOTES ***
        #  What happens when the camera begins acquiring images depends on the
        #  acquisition mode. Single frame captures only a single image, multi
        #  frame catures a set number of images, and continuous captures a
        #  continuous stream of images.
        #
        #  *** LATER ***
        #  Image acquisition must be ended when no more images are needed.
        cam.BeginAcquisition()

        print('Acquiring images for YOLO inference...')

        # Prepare display window using OpenCV for faster rendering
        window_name = 'Thermal YOLO Inference'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while continue_recording:
            try:
                image_result = cam.GetNextImage()

                if image_result.IsIncomplete():
                    print('Image incomplete with image status %d ...' % image_result.GetImageStatus())
                else:
                    # Get numpy array and convert to Celsius
                    image_data = image_result.GetNDArray()
                    image_celsius = image_data * 0.04 - 273.15

                    # Normalize to 0-255 and convert to uint8
                    minv = image_celsius.min()
                    maxv = image_celsius.max()
                    if maxv - minv <= 0:
                        scaled = numpy.zeros_like(image_celsius, dtype=numpy.uint8)
                    else:
                        scaled = ((image_celsius - minv) / (maxv - minv) * 255.0).astype(numpy.uint8)

                    # Convert single-channel thermal to 3-channel BGR for YOLO and visualization
                    bgr = cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)

                    # Resize to model input size while keeping aspect (YOLO will handle letterbox internally in many versions)
                    resized = cv2.resize(bgr, (imgsz, imgsz))

                    # Run inference - using ultralytics model.predict with array input
                    try:
                        results = model.predict(source=resized, conf=conf, save=False, device=device, verbose=False)
                    except TypeError:
                        # Older/newer ultralytics API variations: try predict with different args
                        results = model.predict(resized, conf=conf)

                    # Results is list-like; get first
                    if len(results) > 0:
                        r = results[0]
                    else:
                        r = None

                    vis = resized.copy()

                    # Draw detections if present. The ultralytics result object contains boxes in r.boxes or r.boxes.xyxy
                    try:
                        boxes = []
                        if r is not None:
                            # Try new-style attribute r.boxes
                            if hasattr(r, 'boxes') and r.boxes is not None:
                                for box in r.boxes:
                                    # box.xyxy, box.conf, box.cls
                                    xyxy = box.xyxy.cpu().numpy().astype(int).reshape(-1)
                                    confv = float(box.conf.cpu().numpy().item()) if hasattr(box, 'conf') else 0.0
                                    cls = int(box.cls.cpu().numpy().item()) if hasattr(box, 'cls') else 0
                                    boxes.append((xyxy, confv, cls))
                            elif hasattr(r, 'boxes_xyxy'):
                                # fallback
                                for b in r.boxes_xyxy:
                                    boxes.append((b, 0.0, 0))
                    except Exception:
                        boxes = []

                    # Draw boxes on vis (coordinates are for resized image)
                    for (xyxy, confv, cls) in boxes:
                        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f'{confv:.2f}'
                        cv2.putText(vis, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    # Show the visualization
                    cv2.imshow(window_name, vis)

                    # Handle keypresses: ESC to quit
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27 or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:  # ESC or window closed
                        print('ESC pressed, exiting inference loop')
                        continue_recording = False

                image_result.Release()

            except PySpin.SpinnakerException as ex:
                print('Error: %s' % ex)
                cam.EndAcquisition()
                cv2.destroyAllWindows()
                return False

        cam.EndAcquisition()
        cv2.destroyAllWindows()

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return True

def run_single_camera(cam, infer: bool = True, model_path: str = None, conf: float = 0.25, imgsz: int = 640, device: str = 'cpu'):
    """
    This function acts as the body of the example; please see NodeMapInfo example
    for more in-depth comments on setting up cameras.

    :param cam: Camera to run on.
    :type cam: CameraPtr
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        result = True

        nodemap_tldevice = cam.GetTLDeviceNodeMap()

        # Initialize camera
        cam.Init()

        # Retrieve GenICam nodemap
        nodemap = cam.GetNodeMap()

        # Acquire images and display or infer
        if not infer:
            result &= acquire_and_display_images(cam, nodemap, nodemap_tldevice)
        else:
            result &= acquire_and_infer_yolo(cam, nodemap, nodemap_tldevice, model_path=model_path, conf=conf, imgsz=imgsz, device=device)

        # Deinitialize camera
        cam.DeInit()

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result

def main(args: argparse.Namespace = None):
    """
    Example entry point; notice the volume of data that the logging event handler
    prints out on debug despite the fact that very little really happens in this
    example. Because of this, it may be better to have the logger set to lower
    level in order to provide a more concise, focused log.

    :return: True if successful, False otherwise.
    :rtype: bool
    """
    result = True

    # Retrieve singleton reference to system object
    system = PySpin.System.GetInstance()

    # Get current library version
    version = system.GetLibraryVersion()
    print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

    # Retrieve list of cameras from the system
    cam_list = system.GetCameras()

    num_cameras = cam_list.GetSize()

    print('Number of cameras detected: %d' % num_cameras)

    # Finish if there are no cameras
    if num_cameras == 0:

        # Clear camera list before releasing system
        cam_list.Clear()

        # Release system instance
        system.ReleaseInstance()

        print('Not enough cameras!')
        input('Done! Press Enter to exit...')
        return False

    # Run example on each camera
    for i, cam in enumerate(cam_list):

        print('Running example for camera %d...' % i)

        result &= run_single_camera(cam, infer = args.infer, model_path=args.model_path, conf=args.conf, imgsz=args.imgsz, device=args.device)
        print('Camera %d example complete... \n' % i)

    # Release reference to camera
    # NOTE: Unlike the C++ examples, we cannot rely on pointer objects being automatically
    # cleaned up when going out of scope.
    # The usage of del is preferred to assigning the variable to None.
    del cam

    # Clear camera list before releasing system
    cam_list.Clear()

    # Release system instance
    system.ReleaseInstance()

    input('Done! Press Enter to exit...')
    return result

def parsing():

    parser = argparse.ArgumentParser(description='YOLO Rodent Detection on FLIR AX5 Camera')

    parser.add_argument('--infer', type=bool, default = True,
                        help="Whether to run inference (default: True).")
    parser.add_argument('--model_path', type=str, default ="models/yolo11n-finetuned-best.pt",
                        help="Path to YOLO model (default: 'models/yolo11n-finetuned-best.pt').")
    parser.add_argument('--conf', type=float, default = 0.75,
                        help="Confidence threshold for object detection (default: 0.75).")
    parser.add_argument('--imgsz', type=int, default = 640,
                        help="Inference image size (default: 640).")
    parser.add_argument('--device', type=str, default = 'cpu',
                        help="Device for inference ('cpu' or 'cuda', default: 'cpu').")
    args = parser.parse_args()

    for key, value in args.__dict__.items():
        print(f"{key}: {value}")

    return args

if __name__ == '__main__':
    args = parsing()
    main(args)
