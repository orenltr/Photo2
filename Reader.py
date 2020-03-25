from functools import partial
from numpy import array, asarray, vstack, int_, nonzero, roll, loadtxt
from xml.etree import ElementTree as ET
import json

class LabelFileError(Exception):
    pass


class Reader:
    """
    Handling reading data from Socet GXP files (ipf, gpf, cam and iop) and from sampleme app
    """
    
    @classmethod
    def __GetFileLines(cls, filename):
        '''
        Getting all the lines from a text file and returns them as a list.

        PRIVATE METHOD! DO NOT USE OUTSIDE OF CLASS!

        :param filename: file name and path (string)

        :return: list of all the lines in the file (list of strings)
        '''

        fin = open(filename, 'r')       # open file to read from
        fileLines = fin.readlines()          # read all the lines from the file
        fin.close()                     # close the file

        return fileLines   # return a list containing the lines
    
    @classmethod
    def __ExtractFloatDataFromText(cls, lines):
        '''
        Extracting numeric data from list of strings assuming the data is separated by space

        PRIVATE METHOD! DO NOT USE OUTSIDE OF CLASS!

        :param lines: list of strings containing numeric data

        :return: array of numeric data in float format
        '''

        # If a single string is sent, creates a list with the given string and an empty one
        if isinstance(lines, str):
            temp = []
            temp.append('')
            temp.append(lines)
            lines = temp
        
        # Extracting all numeric data from all the lines and storing them in a list
        data = []
        for line in lines:                  # Going through all the lines
            for t in line.split(' '):       # Going through all the items in a single line
                try:
                    data.append(float(t))   # Trying to convert to float, if succeeded adding to list of float data
                except ValueError:
                    pass
        
        return asarray(data)                # Retunring all converted data as an array
    
    @classmethod
    def __ExtractPointsMetaDataFromText(cls, lines):
        '''
        Extracting points metadata (name, type, etc.) from list of strings

        PRIVATE METHOD! DO NOT USE OUTSIDE OF CLASS!

        :param lines: list of lines to extract the data from (list/ndarray of strings)

        :return: Array of the points' metadata (ndarry)

        '''

        # Splitting the lines based on a space separator
        pntMetaData = list(map(lambda l: l.split(' '), lines))
        
        # Removing the last line in case its blank
        if (pntMetaData[-1][0] == ''):
            pntMetaData = pntMetaData[:-1]
            
        return vstack(pntMetaData) # Return data as ndarray
    
    @classmethod
    def ReadIopFile(cls, filename):
        '''
        Read the sampled points of fiducials from a \*.iop file
        :param filename: The path and filename of the \*.iop file to be read

        :return: An array of the sampled points in pixels (nx2) and another one with
                 the coordinates of the fiducials as they were read from the file (nx2)
        '''

        lines = Reader.__GetFileLines(filename)
        
        numPnts = int(lines[1])
        imgData = Reader.__ExtractFloatDataFromText(lines[4::6])
        fidData = Reader.__ExtractFloatDataFromText(lines[7::6])
        
        return imgData.reshape((numPnts, -1)), fidData.reshape((numPnts, -1))
    
    @classmethod
    def ReadCamFile(cls, filename):
        '''
        Read data from camera file (\*.cam)

        :param filename: The path and filename of the camera file to be read

        :return: A dictionary containing all calibration data with the following keys:

               -  'f' - for the focal length,
               -  'xp' - for the offset of the principal point in the x-axis
               -  'yp' - for the offset of the principal point in the y-axis
               -  'fiducials' - for the list of camera's fiducials
               -  'k0', 'k1', 'k2', 'k3' - for the radial distortions coefficients
               -  'p1', 'p2', 'p3' -  - for the decentering distortions coefficients
        '''

        lines = Reader.__GetFileLines(filename)
        
        focalLength = float(lines[1])
        
        offsets = Reader.__ExtractFloatDataFromText(lines[3])
        xp = offsets[0]
        yp = offsets[1]
        
        numFiducials = int(lines[5])
        fiducialsData = (Reader.__ExtractFloatDataFromText(lines[7 : 7 + numFiducials + 1])).reshape((numFiducials, 2))

        radialDistortionParameters = Reader.__ExtractFloatDataFromText(lines[7 + numFiducials + 1])
        decenteringParameters = Reader.__ExtractFloatDataFromText(lines[7 + numFiducials + 3])

        return {'f': focalLength,
                'xp': xp,
                'yp': yp,
                'fiducials': fiducialsData,
                'k0': radialDistortionParameters[0],
                'k1': radialDistortionParameters[1],
                'k2': radialDistortionParameters[2],
                'k3': radialDistortionParameters[3],
                'p1': decenteringParameters[0],
                'p2': decenteringParameters[1],
                'p3': decenteringParameters[2],
                'p4': 0}
    
    @classmethod
    def ReadGpfFile(cls, filename):
        '''
        Read data from ground point file (\*.gpf)

        :param filename: The path and filename of the ground point file to be read

        :return: Dictionaries of control and tie points with the names of the points as keys
        '''

        # initializations
        headerLines = None
        numPnts = None

        lines = Reader.__GetFileLines(filename)
        headerLinesOptions = [1, 11, 27, 28]

        for option in headerLinesOptions:
            try:
                headerLines = option
                numPnts = int(lines[headerLines])
                break
            except ValueError:
                pass

        if (lines[headerLines + 6] == '\n') or (lines[headerLines + 6] == ''):
            pntMetaData = Reader.__ExtractPointsMetaDataFromText(lines[headerLines + 2::5])
            grdData = (Reader.__ExtractFloatDataFromText(lines[headerLines + 3::5])).reshape((numPnts, -1))
        else:
            pntMetaData = Reader.__ExtractPointsMetaDataFromText(lines[headerLines + 2::8])
            grdData = (Reader.__ExtractFloatDataFromText(lines[headerLines + 3::8])).reshape((numPnts, -1))

        pntNames = pntMetaData[:, 0]
        pntTypes = int_(pntMetaData[:, -1])
        
        # Switching order of coordinates to XYZ
        grdData[:, 0:2] = roll(grdData[:, 0:2], 1, axis = 1)
        
        # Finding the indices of all the points which are 3D control or tie points
        ctrlPnts = nonzero(pntTypes == 3)[0]
        tiePnts = nonzero(pntTypes == 0)[0]

        # Returning the control points and tie points as two separate dictionaries based on the points' names
        return dict(list(zip(pntNames[ctrlPnts], grdData[ctrlPnts]))), dict(list(zip(pntNames[tiePnts], grdData[tiePnts])))
            
        
    @classmethod
    def ReadIpfFile(cls, filename):
        '''
        Read data from image point file (\*.ipf)

        :param filename: The path and filename of the image point file to be read

        :return: Dictionaries of control and tie points with the names of the points as keys
        '''

        lines = Reader.__GetFileLines(filename)
        
        numPnts = int(lines[1])
        pntNames = Reader.__ExtractPointsMetaDataFromText(lines[3::6])[:, 0]
        imgData = Reader.__ExtractFloatDataFromText(lines[4::6])

        # Returning extracted data as a dictionary based on the points' names
        return dict(list(zip(pntNames, imgData.reshape((numPnts, -1)))))

    @classmethod
    def ReadSampleFile(cls, filename):
        """
        Reads sampled points saved from "sampleme" app (\*.json)

        :param filename: path and filename of the json file saved

        :return: an array of the sampled points

        **Example**

        .. code-block:: py

            jsonFileData = Reader.ReadSampleFile('cancel.json')
            print( jsonFileData)
        """
        try:
            with open(filename, 'rb') as f:
                data = json.load(f)
                v=[]
                vertices = ((v['label'], v['point'], v['fill_color']) \
                            for v in data['vertices'])
                for label, point, fill_color in vertices:
                    v.append((point[0], point[1]))

                return asarray(v)

        except Exception as e:
            raise LabelFileError(e)

    @classmethod
    def photoModXMLReader(cls,filename):
        """

        :param filename:
        :return:
        """
        tree = ET.parse(filename)  # getting xml tree
        root = tree.getroot()  # getting root node

        imagesNode = root[-2]  # getting images data node (assumed as second node from the end)
        images = []
        for child in imagesNode:
            imgId = child[0].attrib['v']  # getting image id
            imgName = child[1].attrib['v']  # getting image name
            images.append(array([imgId, imgName]))
        images = array(images)

        pointsNode = root[-1]  # getting points data node (assumed as the last node)
        grdPnts = []
        imgPnts = []
        for child in pointsNode:
            pntAttributes = child[0]
            pntName = pntAttributes[2].attrib['v']  # getting point name
            pntType = pntAttributes[3].attrib['v']  # getting point type
            if pntType == 'control':
                pntGroundCoords = pntAttributes[4]  # getting point ground coordinates
                pntX = float(pntGroundCoords[0].attrib['v'])  # getting X ground coordinate
                pntY = float(pntGroundCoords[1].attrib['v'])  # getting Y ground coordinate
                pntZ = float(pntGroundCoords[2].attrib['v'])  # getting Z ground coordinate

                grdPnts.append(array([pntName, pntType, pntX, pntY, pntZ]))

            pntMeasurements = child[1]  # getting point samples at all visible images
            for _child in pntMeasurements:
                imgId = _child[0].attrib['v']  # getting image id
                imgX = float(_child[1][0].attrib['v'])  # getting image x-coordinate
                imgY = float(_child[1][1].attrib['v'])  # getting image y-coordinate
                imgPnts.append(array([imgId, imgX, imgY]))

        grdPnts = array(grdPnts)
        imgPnts = array(imgPnts)

        return images, grdPnts, imgPnts

    @classmethod
    def Readtxtfile(cls, fileName):
        """
        Read txt file contents into a numpy array

        :param fileName: file name

        :return: an array of the file contents
        """
        data = loadtxt(fname=fileName, skiprows=1)

        return data

if __name__ == "__main__":

    # fidImgData, fidCamData = Reader.ReadIopFile('../data/AR112-3574.iop')
    # camFileData = Reader.ReadCamFile('../data/rc30.cam')
    # print(camFileData['f'], camFileData['xp'], camFileData['yp'])
    # print(camFileData['fiducials'])
    # print(camFileData['k0'], camFileData['k1'], camFileData['k2'], camFileData['k3'])
    # print(camFileData['p1'], camFileData['p2'], camFileData['p3'], camFileData['p4'])
    #
    # imgData = Reader.ReadIpfFile('../data/AR112-3574.ipf')
    # print(imgData.keys())
    #
    # ctrlPnts, tiePnts = Reader.ReadGpfFile('../data/Ground Points 20150531 1.gpf')
    # print(ctrlPnts.keys())

    # jsonFileData = Reader.ReadSampleFile('cancel.json')
    # print( jsonFileData)
    # print(Reader.Readtxtfile('fiducialsImg.txt'))
    images, grdPnts, imgPnts = Reader.photoModXMLReader('exp.x-points')
    print(imgPnts)
