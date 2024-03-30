
import copy  # for deepcopy
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np  
import numpy as np
import random
import math
from collections import Counter
import itertools
from pprint import pprint
import json
from datetime import timedelta, date
from dateutil import rrule
from datetime import datetime, timedelta
import datetime
import time
#from  random import *
import simpy
#from config import *
import networkx as nx
from itertools import islice
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from collections import Counter


from datetime import datetime
import networkx as nx
import math
from itertools import islice
import pandas as pd
import warnings
import time
import yaml
import pprint
import json
from functools import reduce
import operator


NO_OF_SLOT=[1,2,3,4]
RANDOM_SEED = [50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340]
MAX_TIME = 10000000  
DEMAND_SLOT_QPSK=[1,2,3,4]
dem_slot=[2,3,4,5]
DEMAND_TYPE = [1,2,3,4]
TOPOLOGY = 'dutch_telecom'
HOLDING_TIME = 2.0
SLOTS = 360
SLOT_SIZE = 12.5
N_PATH = 1


#Deallocate spectrum after the expiary of holtding time
class Deallocate(object):
	def __init__(self, env):
		self.env = env
	def Run(self, count, path, spectrum, holding_time,partition):
		global topology
		yield self.env.timeout(holding_time)
		for i in range(0, (len(path)-1)):
			for slot in range(spectrum[0],spectrum[1]+1):
				topology[path[i]][path[i+1]][partition][slot] = 0


class Simulador(object):
	def __init__(self, env):
		self.env = env
		global topology
		global topology2
		demand_dict={}


		self.nodes = list(topology.nodes())
		self.random = random.Random()
		self.NumReqBlocked = 0 
		self.cont_req = 0
		self.k_paths = {}
		self.DemandT1=0
		self.DemandT2=0
		self.DemandT3=0
		self.DemandT4=0
		self.DemandT1_Blocked=0
		self.DemandT2_Blocked=0
		self.DemandT3_Blocked=0
		self.DemandT4_Blocked=0
		self.LeftReq=0

		self.slot_usage_count={}
		self.tot_holding_time=0

		self.all_neighboring_links={}

	def Run(self, rate,demand_seq,day,hr,next_slot_dem,iter_dem_pair):
		global topology
		global topology2

		for u, v in list(topology.edges):
			topology[u][v]['d1'] = [0] * next_slot_dem[0]
			topology[u][v]['d2'] = [0] * next_slot_dem[1]
			topology[u][v]['d3'] = [0] * next_slot_dem[2]
			topology[u][v]['d4'] = [0] * next_slot_dem[3]

			self.slot_usage_count[topology2[u][v]['id']]=[0]*4

		for i in list(topology.nodes()):
			for j in list(topology.nodes()):
				if i!= j:
					self.k_paths[i,j] = self.k_shortest_paths(topology, i, j, N_PATH, weight='weight')

		#precompute all the neighboring links of all the paths in the network
		self.all_neighboring_links = self.find_neighboring_links_of_all_paths()
		#print(self.all_neighboring_links)
    
		for count in range((len(demand_seq))):
			yield self.env.timeout(self.random.expovariate(rate))
			#print("-----------------------------------------------")
			src=demand_seq[count][0]
			dst=demand_seq[count][1]
			#print(src,dst)
			bandwidth = self.getBW(src,dst)
			if(bandwidth==0):
				self.LeftReq+=1
				#print("List Empty")
				continue
			demand_type=self.Demand_Type(bandwidth)
			holding_time = self.random.expovariate(HOLDING_TIME)
			self.Count_Demand_Type(demand_type)
			paths = self.k_paths[src,dst]

			flag = 0
			for i in range(N_PATH):
				num_slots=self.Calculte_Num_Slot(demand_type)
				#print("Num Slots:",num_slots)
				free_block=self.PathIsAble_free_blocks(num_slots,paths[i],demand_type)
				#print("Path: ",paths[i])
				if free_block: # free_block has some items

					#print("Free Blocks: ",free_block)
					n_links = self.all_neighboring_links[tuple(paths[i])]
					occupied_block_on_neighborLink=self.check_common_occupied_blocks_block_key(free_block,n_links,demand_type)
					#print(occupied_block_on_neighborLink)

					if(occupied_block_on_neighborLink=={}):
						spectrum_block=[free_block[0][0],free_block[0][1]]
						self.cont_req += 1
						self.FirstFreeFit(count,spectrum_block[0],spectrum_block[1],paths[i],demand_type,holding_time)
						deallocate = Deallocate(self.env) #call Deaalocate Function
						self.env.process(deallocate.Run(count,paths[i],spectrum_block,holding_time,demand_type)) #start the deallocation process with timeout=holding_time Run(connectionID,path[10,3,1,2],[startSlotIndex,EndSlotIndex],holding_time)
						flag = 1 #connection established
						#print("Connection established on First free block ")
						break
					else:
						best_block=self.find_best_block_for_allocation(occupied_block_on_neighborLink,holding_time)
						spectrum_block=[best_block[0],best_block[1]]
						self.cont_req +=1
						self.FirstFreeFit(count,spectrum_block[0],spectrum_block[1],paths[i],demand_type,holding_time)
						deallocate=Deallocate(self.env) #call Deaalocate Function
						self.env.process(deallocate.Run(count,paths[i],spectrum_block,holding_time,demand_type))
						flag=1
						#print("Connection established on best block")
						break
				else:
					break
				

			if flag == 0:
					#print("Connection Blocked")
					self.NumReqBlocked +=1
					# self.Count_Demand_Type_Blocked(demand_type)

		 
	
	
	def find_neighboring_links_of_a_path(self,path):
		global topology
		neighboring_links = set()
		path_edges = set(zip(path, path[1:]))  # Create a set of edges that are part of the path
		# Iterate through each node in the path
		for node_id in path:
			for neighbor in topology.neighbors(node_id):
				if (node_id, neighbor) not in path_edges and (neighbor, node_id) not in path_edges:
					neighboring_links.add((node_id, neighbor))
					neighboring_links.add((neighbor, node_id))
		# Remove any links that directly connect nodes in the path
		for edge in path_edges:
			neighboring_links.discard(edge)
			neighboring_links.discard((edge[1], edge[0]))  # Remove reverse direction if added
		unique_links = [tuple(sorted(set(link))) for link in neighboring_links]
		unique_links = list(set(unique_links))
		return unique_links

	def find_neighboring_links_of_all_paths(self):
		global topology
		neighborLinks = {}  # Dictionary to store neighboring links of all possible paths
		for path_key, path in self.k_paths.items():
			converted_list = [item for sublist in path for item in sublist]
			#print(converted_list)
			neighboring_links = self.find_neighboring_links_of_a_path(converted_list)
			neighborLinks[tuple(converted_list)] = neighboring_links  # Convert path_key to tuple
		return neighborLinks
	

	
	def PathIsAble_free_blocks(self, nslots, path, partition):
		global topology
		
		free_blocks = []  # List to store the start and end of each non-overlapping free block of size nslots
		slot = 0
		max_slot = len(topology[path[0]][path[1]][partition])  # Precompute the maximum slot index
		while slot < max_slot:
			if all(topology[path[ind]][path[ind+1]][partition][slot] == 0 for ind in range(len(path)-1)):
				# Check if the current slot is free across the entire path
				block_start = slot
				while slot < max_slot and all(topology[path[ind]][path[ind+1]][partition][slot] == 0 for ind in range(len(path)-1)):
					slot += 1
				block_end = slot - 1  # slot is now one past the end of the free block
				if block_end - block_start + 1 >= nslots:
					# Check if the free block is large enough for nslots
					free_blocks.append((block_start, block_start + nslots - 1))
				# No need to continue searching if the current block is not large enough
			else:
				slot += 1  # Move to the next slot if the current slot is not free across the entire path
		return free_blocks


	def check_common_occupied_blocks_block_key(self, free_blocks, neighbor_links, partition):
		global topology

		occupied_blocks = {}

		for block in free_blocks:
			block_start, block_end = block
			block_key = (block_start, block_end)  # Use both start and end as the key

			for link in neighbor_links:
				for slot in range(block_start, block_end + 1):
					release_time = topology[link[0]][link[1]][partition][slot]

					if release_time != 0:  # Slot is occupied
						release_time = release_time[1]

						if block_key not in occupied_blocks:
							occupied_blocks[block_key] = {link: {release_time}}
						else:
							if link not in occupied_blocks[block_key]:
								occupied_blocks[block_key][link] = {release_time}
							else:
								occupied_blocks[block_key][link].add(release_time)

		sorted_occupied_blocks = dict(sorted(occupied_blocks.items(), key=lambda item: len(item[1]), reverse=True))
		return sorted_occupied_blocks



	def find_best_block_for_allocation(self, occupied_block_on_neighborLink, ht):
		# Calculate current relative release time
		current_r_rtime = time.time() + ht
		
		# Initialize data list
		data = []
		
		# Iterate over occupied blocks on neighbor links
		for block_key in occupied_block_on_neighborLink:
			# Combine all release times into a single list
			rt_list = [release_time for link_releases in occupied_block_on_neighborLink[block_key].values() for release_time in link_releases]
			rt_list.append(current_r_rtime)  # Append current relative release time
			num_link = len(rt_list)
			variance = np.var(rt_list) if rt_list else 0
			data.append([block_key, rt_list, variance, num_link])
		
		variance_weight = 0.9
		link_count_weight = 0.1
		
		# Initialize variables to track the best block found
		best_block_key = None
		best_score = float('inf')  # Initialize with infinity to ensure first block is selected
		
		# Iterate through the data
		for block_key, _, variance, link_count in data:
			score = variance_weight * variance - link_count_weight * link_count
			if score < best_score:
				best_block_key = block_key
				best_score = score
		
		return best_block_key

	

	
	def FirstFreeFit(self, count, i, j, path, partition, ht):
		global topology
		
		beginning = i  # Start slot index
		end = j  # End slot index
		release_time = time.time() + ht

		# Iterate over each link in the path
		for k in range(len(path) - 1):
			link_start = path[k]
			link_end = path[k + 1]
			
			# Iterate over each slot in the range [beginning, end)
			for slot in range(beginning, end):
				# Assign the connection ID and release time to the slot on the current link
				topology[link_start][link_end][partition][slot] = (count, release_time)


	def getBW(self,  src, dst):
		list_len=len(iter_dem_pair[src,dst])
		if(list_len):
			bw=(iter_dem_pair[src,dst]).pop(random.randrange(list_len))
			return bw
		else:
			return 0
		 
	#Calculates the k-shortest paths between o-d pairs
	def k_shortest_paths(self,G, source, target, k, weight='weight'):
		return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))
	
	#Identify the demand type based on the bandwidth
	def Demand_Type(self, bandwidth):
		if bandwidth==1:
			return 'd1'
		elif bandwidth==2:
			return 'd2'
		elif bandwidth==3:
			return 'd3'
		else:
			return 'd4'
		
	#compute no of slot required for each demand type using QPSk Modulation Scheme
	def Calculte_Num_Slot(self,demand):
		if demand=='d1':
			num_slots=2
		elif demand=='d2':
			num_slots=3
		elif demand=='d3':
			num_slots=4
		else:
			num_slots=5
		return num_slots




