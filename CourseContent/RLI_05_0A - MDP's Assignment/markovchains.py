# Adapted from https://github.com/NaysanSaran/markov-chain
"""
Markov Chain Transition Diagrams
Simple Markov Chain visualization module in Python. 
Requires matplotlib and numpy to work.
"""
import numpy as np
import matplotlib.patches as mpatches 
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt

# module from this repository
# from node import Node
class Node():
    
    def __init__(
        self, center, radius, label, 
        facecolor='#2693de', edgecolor='#e6e6e6',
        ring_facecolor='#a3a3a3', ring_edgecolor='#a3a3a3'
        ):
        """
        Initializes a Markov Chain Node(for drawing purposes)
        Inputs:
            - center : Node (x,y) center
            - radius : Node radius
            - label  : Node label
        """
        self.center = center
        self.radius = radius
        self.label  = label

        # For convinience: x, y coordinates of the center
        self.x = center[0]
        self.y = center[1]
        
        # Drawing config
        self.node_facecolor = facecolor
        self.node_edgecolor = edgecolor
        
        self.ring_facecolor = ring_facecolor
        self.ring_edgecolor = ring_edgecolor
        self.ring_width = 0.03  
        
        self.text_args = {
            'ha': 'center', 
            'va': 'center', 
            'fontsize': 16
        }
    
    
    def add_circle(self, ax):
        """
        Add the annotated circle for the node
        """
        circle = mpatches.Circle(self.center, self.radius)
        p = PatchCollection(
            [circle], 
            edgecolor = self.node_edgecolor, 
            facecolor = self.node_facecolor
        )
        ax.add_collection(p)
        ax.annotate(
            self.label, 
            xy = self.center, 
            color = '#ffffff', 
            **self.text_args
        )
        
        
    def add_self_loop(self, ax, prob=None, direction='up'):
        """
        Draws a self loop
        """
        if direction == 'up':
            start = -30
            angle = 180
            ring_x = self.x
            ring_y = self.y + self.radius
            prob_y = self.y + 1.3*self.radius
            x_cent = ring_x - self.radius + (self.ring_width/2)
            y_cent = ring_y - 0.15
        else:
            start = -210
            angle = 0
            ring_x = self.x
            ring_y = self.y - self.radius
            prob_y = self.y - 1.4*self.radius
            x_cent = ring_x + self.radius - (self.ring_width/2)
            y_cent = ring_y + 0.15
            
        # Add the ring
        ring = mpatches.Wedge(
            (ring_x, ring_y), 
            self.radius, 
            start, 
            angle, 
            width = self.ring_width
        )
        # Add the triangle (arrow)
        offset = 0.2
        left   = [x_cent - offset, ring_y]
        right  = [x_cent + offset, ring_y]
        bottom = [(left[0]+right[0])/2., y_cent]
        arrow  = plt.Polygon([left, right, bottom, left])

        p = PatchCollection(
            [ring, arrow], 
            edgecolor = self.ring_edgecolor, 
            facecolor = self.ring_facecolor
        )
        ax.add_collection(p)
        
        # Probability to add?
        if prob:
            ax.annotate(str(round(prob,2)).lstrip('0'), xy=(self.x, prob_y), color='#000000', **self.text_args)


class MarkovChain:

    def __init__(self, M, labels, pos=[], colors=[]):
        """
        Initializes a Markov Chain (for drawing purposes)
        Inputs:
            - M         Transition Matrix
            - labels    State Labels
            - pos       Positions of nodes
            - colors    Colors for nodes
        """

        if M.shape[0] < 2:
            raise Exception("There should be at least 2 states")
        if M.shape[0] != M.shape[1]:
            raise Exception("Transition matrix should be square")
        if M.shape[0] != len(labels):
            raise Exception("There should be as many labels as states")

        self.M = M
        self.n_states = M.shape[0]
        self.labels = labels

        # Colors
        self.arrow_facecolor = '#a3a3a3'
        self.arrow_edgecolor = '#a3a3a3'

        self.node_facecolor = '#2693de'
        self.node_edgecolor = '#e6e6e6'

        # Drawing config
        self.node_radius = 0.5
        self.arrow_width = 0.03
        self.arrow_head_width = 0.20
        self.text_args = {
            'ha': 'center',
            'va': 'center',
            'fontsize': 16
        }

        # Build the network
        self.build_network(pos,colors)


    def set_node_centers(self,pos=[]):
        # Node positions
        if pos == []:
            self.node_centers = []
            for k in range(self.n_states):
                alpha = -np.pi + 2*np.pi/self.n_states * k
                self.node_centers.append([4*np.cos(alpha), 4*np.sin(alpha)])
        elif pos == 'linear':
            self.node_centers = []
            for k in range(self.n_states):
                self.node_centers.append([2.5*k,0])
        else:
            if len(pos) != len(self.labels):
                raise Exception("There should be as many positions as states")
            self.node_centers =  pos

        self.node_centers = np.array(self.node_centers)
        self.xlim = (min(self.node_centers[:,0])-1, max(self.node_centers[:,0])+1)
        self.ylim = (min(self.node_centers[:,1])-1, max(self.node_centers[:,1])+1)
        self.figsize = (self.xlim[1]-self.xlim[0], self.ylim[1]-self.ylim[0])


    def build_network(self,pos=[],colors=[]):
        """
        Loops through the matrix, add the nodes
        """
        # Position the node centers
        self.set_node_centers(pos)

        # Set the nodes
        self.nodes = []
        for i in range(self.n_states):
            color = self.node_facecolor if len(colors) <= i else colors[i]
            node = Node(
                self.node_centers[i],
                self.node_radius,
                self.labels[i],
                facecolor = color
            )
            self.nodes.append(node)


    def add_arrow(self, ax, node1, node2, prob=None):
        """
        Add a directed arrow between two nodes
        """
        # x,y start of the arrow
        x_start = node1.x + np.sign(node2.x-node1.x) * node1.radius
        y_start = node1.y + np.sign(node2.y-node1.y) * node1.radius
        # CCW counter clock wise alpha
        alpha = np.arctan2(node2.y-node1.y,node2.x-node1.x)
        x_start = node1.x + node1.radius * np.cos(alpha + np.pi/10)
        y_start = node1.y + node1.radius * np.sin(alpha + np.pi/10)
        x_end = node2.x + node2.radius*1.5 * np.cos(alpha - np.pi/10 + np.pi)
        y_end = node2.y + node2.radius*1.5 * np.sin(alpha - np.pi/10 + np.pi)
        dx = x_end - x_start
        dy = y_end - y_start
        
        arrow = mpatches.FancyArrow(
            x_start,
            y_start,
            dx,
            dy,
            width = self.arrow_width,
            head_width = self.arrow_head_width
        )
        p = PatchCollection(
            [arrow],
            edgecolor = self.arrow_edgecolor,
            facecolor = self.arrow_facecolor
        )
        ax.add_collection(p)
        # Probability to add?
        x_prob = x_start + 0.2*dx
        y_prob = y_start + 0.2*dy
        if prob:
            ax.annotate(str(round(prob,2)).lstrip('0'), xy=(x_prob, y_prob), color='#000000', **self.text_args)


    def draw(self, img_path=None):
        """
        Draw the Markov Chain
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Set the axis limits
        plt.xlim(self.xlim)
        plt.ylim(self.ylim)

        # Draw the nodes
        for node in self.nodes:
            node.add_circle(ax)

        # Add the transitions
        for i in range(self.M.shape[0]):
            for j in range(self.M.shape[1]):
                p = self.M[i,j]
                # self loops
                if i == j and p > 0:
                    # Loop direction
                    if self.nodes[i].y >= 0:
                        self.nodes[i].add_self_loop(ax, prob = p, direction='up')
                    else:
                        self.nodes[i].add_self_loop(ax, prob = p, direction='down')
                # directed arrows
                elif p > 0:
                    self.add_arrow(ax, self.nodes[i], self.nodes[j], prob = p)

        plt.axis('off')
        # Save the image to disk?
        if img_path:
            plt.savefig(img_path)
        plt.show()