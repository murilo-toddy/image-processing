#Assignment 5
#Morphology and Image Description
#Lucas Toschi de Oliveira
#USP ID: 11234190

#import matplotlib.pyplot as plt
import numpy as np
import imageio
import random

import time

#Show all images side by side for comparation.
# def showImages(images, side_size):
#     plt.figure(figsize=(len(images)*side_size, side_size))

#     subplotIndex = int("1" + str(len(images)) + "1")
#     for image in images:
#         plt.subplot(subplotIndex)
#         plt.imshow(image, interpolation="none")
#         subplotIndex += 1
#     plt.show()

#Calculate and returns the RSE error between the original image and the reference image.
def getRSE(reference_image, output_image):
    if len(output_image.shape) == 3:
        total_error = 0
        for i in range(0, output_image.shape[2]):
            error_sum = np.sum((reference_image[:, :, i] - output_image[:, :, i])**2)
            error_average = error_sum/(reference_image.shape[0]*reference_image.shape[1])
            total_error += round(np.sqrt(error_average), 4)
            return round(total_error/3, 4)
    else:
        error_sum = np.sum((reference_image - output_image)**2)
        error_average = error_sum/(reference_image.shape[0]*reference_image.shape[1])
        return round(np.sqrt(error_average), 4)
    
def resetClusters(initialClusters, clusters):
    for i in range(0, len(clusters)):
        clusters[i]["x"] = initialClusters[i]["x"]
        clusters[i]["y"] = initialClusters[i]["y"]


#Normalize an image with to an arbitrary bit resolution.
def normalizeImage(image, bit_resolution):
    max_value = np.max(image)
    min_value = np.min(image)

    image = (image - min_value)*((2**bit_resolution - 1)/(np.abs(max_value - min_value)))

    return image

def buildImageFromClusters(parameters, input_image, clusters):
    outputImage = []

    if parameters["pixel_attributes_option"] <= 2: 
        outputImage = np.zeros(input_image.shape)
    
        for cluster in clusters:
            for i in range(0, len(cluster["x"])):
                outputImage[cluster["x"][i], cluster["y"][i], 0] = int(cluster["R"])
                outputImage[cluster["x"][i], cluster["y"][i], 1] = int(cluster["G"])
                outputImage[cluster["x"][i], cluster["y"][i], 2] = int(cluster["B"])

        outputImage[:, :, 0] = normalizeImage(outputImage[:, :, 0], 8)
        outputImage[:, :, 1] = normalizeImage(outputImage[:, :, 1], 8)
        outputImage[:, :, 2] = normalizeImage(outputImage[:, :, 2], 8)
    else:
        outputImage = np.zeros([input_image.shape[0], input_image.shape[1]])
        
        for cluster in clusters:
            for i in range(0, len(cluster["x"])):
                outputImage[cluster["x"][i], cluster["y"][i]] = int(cluster["luminance"])

        outputImage = normalizeImage(outputImage, 8)

    return outputImage


#Converts the images vector from RGB to gray images using Luminance weights.
#Returns the vector with the transformation applied in each image.
def rgbToGray(rgbValues): 
    return (0.299*rgbValues[0] + 0.587*rgbValues[1] + 0.114*rgbValues[2])

def getRGBEuclidianDistance(reference_data, pixel_data):
    delta2R = pow(reference_data["R"] - pixel_data["R"], 2)
    delta2G = pow(reference_data["G"] - pixel_data["G"], 2)
    delta2B = pow(reference_data["B"] - pixel_data["B"], 2)

    return np.sqrt(delta2R + delta2G + delta2B)

def getRGBxyEuclidianDistance(reference_data, pixel_data):
    delta2R = pow(reference_data["R"] - pixel_data["R"], 2)
    delta2G = pow(reference_data["G"] - pixel_data["G"], 2)
    delta2B = pow(reference_data["B"] - pixel_data["B"], 2)
    delta2X = pow(reference_data["x_med"] - pixel_data["x"], 2)
    delta2Y = pow(reference_data["y_med"] - pixel_data["y"], 2)

    return np.sqrt(delta2R + delta2G + delta2B + delta2X + delta2Y)

def getLuminanceEuclidianDistance(reference_data, pixel_data):
    return np.sqrt(pow(reference_data["luminance"] - pixel_data["luminance"], 2))

def getLuminanceXYEuclidianDistance(reference_data, pixel_data):
    delta2Lum = pow(reference_data["luminance"] - pixel_data["luminance"], 2)
    delta2X = pow(reference_data["x_med"] - pixel_data["x"], 2)
    delta2Y = pow(reference_data["y_med"] - pixel_data["y"], 2)

    return np.sqrt(delta2Lum + delta2X + delta2Y)


def getEuclidianDistance(parameters, reference_data, pixel_data):
    pixel_attributes_option = parameters["pixel_attributes_option"]

    attributes_method_mapping = {
        1: getRGBEuclidianDistance,
        2: getRGBxyEuclidianDistance,
        3: getLuminanceEuclidianDistance,
        4: getLuminanceXYEuclidianDistance
    }

    return attributes_method_mapping[pixel_attributes_option](reference_data, pixel_data)

def updateCluster(clusters, betterClusterIndex, pixel_data):
    clusters[betterClusterIndex]["x"].append(pixel_data["x"])
    clusters[betterClusterIndex]["y"].append(pixel_data["y"])
    
    if parameters["pixel_attributes_option"] <= 2: 
        clusters[betterClusterIndex]["R_sum"] += pixel_data["R"]
        clusters[betterClusterIndex]["G_sum"] += pixel_data["G"]
        clusters[betterClusterIndex]["B_sum"] += pixel_data["B"]

        clusters[betterClusterIndex]["R"] = clusters[betterClusterIndex]["R_sum"]/float(len(clusters[betterClusterIndex]["x"]))
        clusters[betterClusterIndex]["G"] = clusters[betterClusterIndex]["G_sum"]/float(len(clusters[betterClusterIndex]["x"]))
        clusters[betterClusterIndex]["B"] = clusters[betterClusterIndex]["B_sum"]/float(len(clusters[betterClusterIndex]["x"]))
    else:
        clusters[betterClusterIndex]["luminance_sum"] += pixel_data["luminance"]
        clusters[betterClusterIndex]["luminance"] = clusters[betterClusterIndex]["luminance_sum"]/float(len(clusters[betterClusterIndex]["x"]))

    if parameters["pixel_attributes_option"] % 2 == 0:
        clusters[betterClusterIndex]["x_sum"] += pixel_data["x"]
        clusters[betterClusterIndex]["y_sum"] += pixel_data["y"]

        clusters[betterClusterIndex]["x_med"] = clusters[betterClusterIndex]["x_sum"]/float(len(clusters[betterClusterIndex]["x"]))
        clusters[betterClusterIndex]["y_med"] = clusters[betterClusterIndex]["y_sum"]/float(len(clusters[betterClusterIndex]["x"]))

#Assign each pixel to the cluster relative to the centroid with the smallest distance to the pixel.
def assignPixelsAndUpdateClusters(parameters, images, clusters):   
    input_image = images["input"]
    
    for x in range(0, input_image.shape[0]):
        for y in range(0, input_image.shape[1]):
            pixel_data = {
                "x": x,
                "y": y
            }

            rgb = input_image[x, y]
            if parameters["pixel_attributes_option"] <= 2: 
                pixel_data["R"] = rgb[0]
                pixel_data["G"] = rgb[1]
                pixel_data["B"] = rgb[2]
            else:
                pixel_data["luminance"] = rgbToGray(rgb)

            minorDistance = np.Infinity
            betterClusterIndex = 0
            t0 = time.time()
            for i in range(0, len(clusters)):
                distance = getEuclidianDistance(parameters, clusters[i], pixel_data)
                if distance < minorDistance:
                    minorDistance = distance
                    betterClusterIndex = i

            t1 = time.time()
            print(f"time {t1 - t0}")
            updateCluster(clusters, betterClusterIndex, pixel_data)
            

#Initialize clusters for the input image.
#Returns 
def initializeClusters(parameters, images):
    m, n, _ = images["input"].shape
    k = parameters["clusters_number"]
    
    random.seed(parameters["seed"])
    clustersIndices = np.sort(random.sample(range(0, m*n), k))

    clustersCentroid = []
    for index in clustersIndices:
        x = int(index/m)
        if index % n == 0:
            y = 0
        else:
            y = int(index % n) - 1

        clusterInfo = {
            "x": [x],
            "y": [y],
            "x_med": x,
            "y_med": y,
            "x_sum": x,
            "y_sum": y
        }

        rgb = images["input"][x, y]
        if parameters["pixel_attributes_option"] <= 2: 
            clusterInfo["R"] = rgb[0]
            clusterInfo["R_sum"] = rgb[0]
            clusterInfo["G"] = rgb[1]
            clusterInfo["G_sum"] = rgb[1]
            clusterInfo["B"] = rgb[2]
            clusterInfo["B_sum"] = rgb[2]
        else:
            luminance = rgbToGray(rgb)
            clusterInfo["luminance"] = luminance
            clusterInfo["luminance_sum"] = luminance 

        clustersCentroid.append(clusterInfo)

    return clustersCentroid

#Reads all images needed to run the assignment.
#Returns a Python dictionary with the images. 
def readAllImages(parameters):
    images = {}
    images["input"] = imageio.imread(parameters["input_filename"]).astype(np.float64)
    images["reference"] = imageio.imread(parameters["reference_filename"]).astype(np.float64)

    return images

#Get all parameters needed to run the assignment.
#Returns a Python dictionary with the parameters.
def getAllParameters():
    parameters = {}
    parameters["input_filename"] = str(input().rstrip())
    parameters["reference_filename"] = str(input().rstrip())
    parameters["pixel_attributes_option"] = int(input().rstrip())
    parameters["clusters_number"] = int(input().rstrip())
    parameters["iterations_number"] = int(input().rstrip())
    parameters["seed"] = int(input().rstrip())

    return parameters


if __name__ == "__main__":
    parameters = getAllParameters()
    images = readAllImages(parameters)
    initialClusters = initializeClusters(parameters, images)

    clusters = initialClusters
    for _ in range(parameters["iterations_number"]):
        assignPixelsAndUpdateClusters(parameters, images, clusters)
        resetClusters(initialClusters, clusters)
    
    outputImage = buildImageFromClusters(parameters, images["input"], clusters).astype(np.uint8)
    #showImages([outputImage], 6)
    print(getRSE(images["reference"], outputImage))
