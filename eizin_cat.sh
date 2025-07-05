#!/usr/bin/env bash


ls -ltr gallery/gallery.eizin.co.jp/ | grep 'index.html?p=' | awk '{print "gallery/gallery.eizin.co.jp/" $NF}' | xargs ./eizin_catalog.py
