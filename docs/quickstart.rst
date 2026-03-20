Quickstart
==========

Here is a quick example of how to use **flaretorch** to train a solar flare forecasting model.

.. code-block:: python

   import flaretorch
   from flaretorch.models import ResNetMCD
   from flaretorch.data import FlareDataModule

   # Initialize data module
   dm = FlareDataModule(data_dir="data/")

   # Initialize model
   model = ResNetMCD(input_channels=1, num_classes=2)

   # Train model
   trainer = flaretorch.Trainer(max_epochs=10)
   trainer.fit(model, dm)
