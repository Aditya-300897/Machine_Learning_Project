#!/usr/bin/env python
# coding: utf-8

# ## Emotion recognition through EEG analysis using ML algorithms
# 
# The following notebook includes the code for the work by the team AL2021 ( Aditya Salvi - u3230853, Lilit Griggs-Atoyan - u3182730).
# Content for this notebook is as follows:

# Table of Contents
#  
# * [Section 1. Data Description](#section1)
#     * [Section 1.1. Data import and description](#section_1_1)
#     * [Section 1.2. Data exploration and manipulation](#section_1_2)
#         * [Section 1.2.1](#section_1_2_1)
#         * [Section 1.2.2](#section_1_2_2)
#         * [Section 1.2.3](#section_1_2_3)
# * [Chapter 2](#chapter2)
#     * [Section 2.1](#section_2_1)
#     * [Section 2.2](#section_2_2)

# 
# 
# #### Section 1.1 <a class="anchor" id="section_1_1"></a>
# 
# #### Section 1.2 <a class="anchor" id="section_1_2"></a>
# 
# ##### Section 1.2.1 <a class="anchor" id="section_1_2_1"></a>
# 
# ##### Section 1.2.2 <a class="anchor" id="section_1_2_2"></a>
# 
# ##### Section 1.2.3 <a class="anchor" id="section_1_2_3"></a>
# 
# ### Section 2 <a class="anchor" id="chapter2"></a>
# 
# #### Section 2.1 <a class="anchor" id="section_2_1"></a>
# 
# #### Section 2.2 <a class="anchor" id="section_2_2"></a>

# ### Section 1. Data Description <a class="anchor" id="section1"></a>

# Firstly, let's import data. 
# For that we need to load some libraries for setting up a working directory (os), data manipulation (pandas), and supporting operations with arrays (numpy).

# In[15]:


import os 
import sys
import pandas as pd
import numpy as np


# In[16]:


#dirname = os.path.dirname(__file__)

os.chdir(os.path.dirname(sys.argv[0]))


# In[19]:


os.chdir(os.path.dirname(os.path.abspath(__file__)))


# In[17]:


os.getcwd()


# In[6]:


script_dir = os.path.dirname(__AL2021 Project Code__)

