# Python3 script for the SED-Builder.
# Use it by calling from a python shell:

#import run_sedbuilder_script
#from run_sedbuilder_script import RunSedBuilder
#run = RunSedBuilder("your_user", "your_password", "your_filename_with_source_name.csv")
#run.setup_method()
#run.upload_file()

import pytest
import time
import json
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

class RunSedBuilder:
  """ Class with functions for run SED-Builder from the shell
  """
  def __init__(self, user, password, filename):
    """ Class constructor:
        user, password and filename shall be passed as string"""
    self.user = user
    self.password = password
    self.filename = os.path.join(os.path.abspath(os.getcwd()),filename)
 
  def setup_method(self):
    """ Setup of the browser"""
    self.driver = webdriver.Firefox()
    self.vars = {}
  
  def teardown_method(self):
    """ Function for close the browser and the run"""
    self.driver.quit()
  
  def upload_file(self):
    # open https://tools.ssdc.asi.it/SED/index.jsp and setWindowSize 1920x1080 
    self.driver.get("https://tools.ssdc.asi.it/SED/index.jsp")
    #self.driver.set_window_size(1920, 1080)
    # login into the webpage
    self.driver.find_element(By.LINK_TEXT, "Login").click()
    self.driver.find_element(By.ID, "username").click()
    self.driver.find_element(By.ID, "username").send_keys(self.user)
    self.driver.find_element(By.ID, "password").click()
    self.driver.find_element(By.ID, "password").send_keys(self.password)
    # load the .csv file in the form (the file must be a list of source names) 
    self.driver.find_element(By.CSS_SELECTOR, "input:nth-child(3)").click()
    self.driver.find_element(By.LINK_TEXT, "List of Sources").click()
    self.driver.find_element(By.ID, "sourceListFile").send_keys(self.filename)
    self.driver.find_element(By.ID,"csvFormat").click()
    # select option ofn list of coordinate (comment the following line if a list of names is passed)
    self.driver.find_element(By.XPATH,"//select[@name='csvFormat']/option[text()='RA, Dec (J2000, degrees)']").click()
    self.driver.find_element(By.ID, "btnUpload").click()
    # deselect external catalog from the right collapsed menu
    self.driver.find_element(By.XPATH, "//body/div[2]/div/div/div[2]/div/div/div[2]/div/div/div[1]/h5/button").click()
    # submit the query and ask for the SED data of the sources
    self.driver.find_element(By.XPATH, "//*[@id='allboxint_1']").click()
    element = self.driver.find_element_by_id('buttonSubmit')
    self.driver.execute_script("arguments[0].click();", element)
    time.sleep(150)
    self.driver.quit()
