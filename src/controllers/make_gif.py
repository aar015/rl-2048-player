import cv2
import numpy
import imageio
from game_logic.play import play


def makeGif(agent, gif_file, num_trials=10, board_size=4, graphic_size=750, top_margin=40, seperator_width=12):
    bestFinalScore = 0
    for i in range(num_trials):
        finalScore, log = play(agent, verbose=True) 
        if finalScore > bestFinalScore:
            bestFinalScore = finalScore
            bestLog = log
    with imageio.get_writer(gif_file, mode='I') as writer:
        for i in range(numpy.shape(bestLog)[0]):
            img=makeImage(bestLog[i][0], bestLog[i][1], board_size, graphic_size,top_margin, seperator_width)
            writer.append_data(img)
            if i == numpy.shape(bestLog)[0]-1:
                for i in range(50):
                    writer.append_data(img)

def makeImage(score, state, board_size=4, graphic_size=750, top_margin=40, seperator_width=12):
    img = numpy.full((graphic_size + top_margin, graphic_size, 3), 255, numpy.uint8)
    # Define colors
    background_color = (146, 135, 125)
    color = {0:(158, 148, 138), 1:(238, 228, 218), 2:(237, 224, 200), 3:(242, 177, 121), 
             4:(245, 149, 99), 5:(246, 124, 95), 6:(246, 94, 59), 7:(237, 207, 114), 
             8:(237, 204, 97), 9:(237, 200, 80), 10:(237, 197, 63), 11:(237, 197, 63),
             12:(62, 237, 193), 13:(62, 237, 193), 14:(62,64,237), 15:(140,62,237)}
    #Set font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Define spacing of tiles
    spacing = int((graphic_size-seperator_width)/board_size)
    # Write score at top of screen
    text = 'The score is ' + str(score)
    textsize = cv2.getTextSize(text, font, 1, 3)[0]
    cv2.putText(img,text,(int((graphic_size-textsize[0])/2),int((3*top_margin/4+textsize[1])/2)),
                font,1,(0,0,0),3,cv2.LINE_AA)
    # Draw squares
    for i in range(4):
        for k in range(4):
            cv2.rectangle(img,
                          (int(seperator_width/2)+k*spacing,int(top_margin+seperator_width/2)+i*spacing),
                          (int(seperator_width/2)+(k+1)*spacing,int(top_margin+seperator_width/2)+(i+1)*spacing),
                          color[state[i][k]], -1)
            if state[i][k] == 0:
                text = ''
            else:
                text = str(2**state[i][k])
            textsize = cv2.getTextSize(text, font, 1.5, 6)[0]
            cv2.putText(img,text,(int(seperator_width/2+k*spacing+(spacing-textsize[0])/2),
                                  int(top_margin+seperator_width/2+i*spacing+(spacing+textsize[1])/2)),
                        font,1.5,(0,0,0),6,cv2.LINE_AA)
            cv2.putText(img,text,(int(seperator_width/2+k*spacing+(spacing-textsize[0])/2),
                                  int(top_margin+seperator_width/2+i*spacing+(spacing+textsize[1])/2)),
                        font,1.5,(255,255,255),2,cv2.LINE_AA)
    # Draw outline grid
    for i in range(5):
        cv2.line(img, 
                 (int(seperator_width/2)+i*spacing,int(top_margin+seperator_width/2)),
                 (int(seperator_width/2)+i*spacing,int(graphic_size+top_margin-seperator_width/2)), 
                 background_color, seperator_width)
    for i in range(5):
        cv2.line(img,
                 (int(seperator_width/2),int(top_margin+seperator_width/2)+i*spacing),
                 (int(graphic_size-seperator_width/2),int(top_margin+seperator_width/2)+i*spacing),
                 background_color,seperator_width)
    return img
