from paper2cmap import Paper2CMap

paper2cmap = Paper2CMap(model_name="gpt-3.5-turb")
paper2cmap.load(r"C:\Users\limmi\Desktop\Ming En\Masters\Modules\Sem 3\CS5260\Project\pdf\s41467-024-45563-x.pdf")
paper2cmap.generate_cmap()
print("Concept map generated successfully.")