#!/usr/bin/python
# -*- coding: utf-8 -*-
import subprocess
import time
from pathlib import Path

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver import ActionChains

from semantic_modeling.config import get_logger, config
from semantic_modeling.data_io import get_ontology
from transformation.models.data_table import DataTable
from transformation.r2rml.commands.modeling import SetSemanticTypeCmd, SetInternalLinkCmd
from transformation.r2rml.r2rml import R2RML


def delay():
    time.sleep(0.5)
    return 0.5

def short_delay():
    time.sleep(0.25)
    return 0.25


def get_recent_noti(idx: int=0):
    total_wait_seconds = 0
    while total_wait_seconds < 10:
        total_wait_seconds += delay()

        # check if load success
        try:
            notis = driver.find_elements_by_css_selector("div.sticky-note")
            if len(notis) == 0:
                continue

            return notis[idx].get_attribute("innerText").strip()
        except NoSuchElementException:
            pass
    else:
        raise TimeoutError()


def remove_all_noti():
    notis = list(driver.find_elements_by_css_selector("div.sticky-queue > div.sticky"))
    for el in notis:
        driver.execute_script("var element = arguments[0]; element.parentNode.removeChild(element);", el)
    delay()

    assert len(driver.find_elements_by_css_selector("div.sticky-queue > div.sticky")) == 0

def upload_source(driver: webdriver.Firefox, file: Path):
    """Upload file to Karma"""
    driver.find_elements_by_css_selector("ul.nav > li.dropdown")[0].click()
    short_delay()

    file_input = driver.find_element_by_css_selector("form#fileupload input")
    driver.execute_script('''
arguments[0].style = ""; 
arguments[0].style.display = "block"; 
arguments[0].style.visibility = "visible";''', file_input)
    file_input.send_keys(str(file))

    delay()
    # select file format
    driver.find_element_by_css_selector("#btnSaveFormat").click()

    delay()
    # select #objects to import
    driver.find_element_by_css_selector("#btnSaveOptions").click()

    total_wait_seconds = 0
    while total_wait_seconds < 30:
        total_wait_seconds += delay()

        # check if worksheet is loaded
        try:
            if driver.find_element_by_css_selector("#WorksheetOptionsDiv a").text.strip() == file.name:
                break
        except NoSuchElementException:
            pass
    else:
        raise Exception("Cannot load worksheet of source: %s" % file.name)

    delay()

def apply_r2rml(driver: webdriver.Firefox, model_file: Path):
    # the upload menu needs to be visible first
    driver.find_element_by_css_selector("#WorksheetOptionsDiv").click()
    short_delay()

    apply_r2rml_menu = driver.find_elements_by_css_selector("#WorksheetOptionsDiv > ul.dropdown-menu li.dropdown-submenu")[1]
    ActionChains(driver).move_to_element(apply_r2rml_menu).perform()
    short_delay()

    file_input = apply_r2rml_menu.find_element_by_css_selector("ul.dropdown-menu > li form > input")
    file_input.send_keys(str(model_file))
    delay()

    try:
        noti = get_recent_noti()
        assert noti == "Model successfully applied!", "Apply R2RML but have error: %s" % noti
    except TimeoutError:
        raise Exception("Apply R2RML timeout")

    remove_all_noti()

def export_r2rml_model(driver: webdriver.Firefox, model_file: Path, replace: bool):
    global logger

    # the upload menu needs to be visible first
    driver.find_element_by_css_selector("#WorksheetOptionsDiv").click()
    short_delay()

    publish_r2rml_menu = driver.find_elements_by_css_selector("#WorksheetOptionsDiv > ul.dropdown-menu li.dropdown-submenu")[2]
    ActionChains(driver).move_to_element(publish_r2rml_menu).perform()
    short_delay()

    publish_btn = publish_r2rml_menu.find_elements_by_css_selector("li")[1]
    assert publish_btn.text.strip() == "Model"
    publish_btn.click()
    delay()

    try:
        noti = get_recent_noti(idx=1)
        assert noti == "R2RML Model published", "Having error while exporting R2RML: %s" % noti
    except TimeoutError:
        raise Exception("Export R2RML Timeout")

    remove_all_noti()
    short_delay()

    if replace:
        download_link = driver.find_element_by_css_selector("a.R2RMLDownloadLink").get_attribute("href")
        subprocess.check_call(["wget", download_link, "-O", str(model_file)])
    delay()

def remove_worksheet(driver: webdriver.Firefox):
    driver.find_element_by_css_selector("#WorksheetOptionsDiv").click()
    short_delay()

    delete_worksheet = driver.find_elements_by_css_selector("#WorksheetOptionsDiv > ul.dropdown-menu > li")[-3]
    assert delete_worksheet.text.strip() == "Delete Worksheet"
    delete_worksheet.click()
    short_delay()

    alert = driver.switch_to.alert
    alert.accept()
    delay()

    remove_all_noti()


# SETUP hyper-parameters
logger = get_logger("app.preprocessing.generate_r2rml")
dataset = "museum_edm"
ont = get_ontology(dataset)

#%% INIT SELENIUM

driver = webdriver.Firefox()
driver.get("http://localhost:8080")
time.sleep(5)

#%% LOAD FILES
model_dir = Path(config.datasets[dataset].models_y2rml.as_path())
r2rml_dir = Path(config.datasets[dataset].as_path()) / "karma-version" / "models-r2rml"
karma_source_dir = Path(config.datasets[dataset].as_path()) / "karma-version" / "sources"

try:
    for file in sorted(karma_source_dir.iterdir()):
        # if not file.name.startswith("s01"):# and not file.name.startswith("s10"):
        #     continue

        logger.info("Generate R2RML for source: %s", file.name)

        r2rml_file = r2rml_dir / f"{file.stem}-model.ttl"
        tbl = DataTable.load_from_file(file)
        r2rml = R2RML.load_from_file(model_dir / f"{file.stem}-model.yml")
        # note that we use a cleaned data table, whatever columns need to create/transform have been done.
        # therefore, we will remove all command that aren't SetSemanticType or SetInternalLink
        r2rml.commands = [cmd for cmd in r2rml.commands if isinstance(cmd, (SetSemanticTypeCmd, SetInternalLinkCmd))]
        r2rml.to_kr2rml(ont, tbl, r2rml_file)

        upload_source(driver, file)
        apply_r2rml(driver, r2rml_file)
        export_r2rml_model(driver, r2rml_file, replace=False)
        remove_worksheet(driver)

        time.sleep(1)
finally:
    pass
    time.sleep(5)
    driver.close()