.PHONY: setup deploy serve

setup:
	bash setup.sh

deploy:
	modal deploy modal_app.py

serve:
	modal serve modal_app.py
