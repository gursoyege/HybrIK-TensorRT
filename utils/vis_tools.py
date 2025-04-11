# This is originally taken from https://github.com/Deep-Imaging-Group/SDSR-OCT/ and modified.

import time
import numpy as np
import torch as t
import visdom
from matplotlib import pyplot as plot

class Visualizer(object):
    """
    wrapper for visdom
    you can still access naive visdom function by 
    self.line, self.scater,self._send,etc.
    due to the implementation of `__getattr__`
    """

    def __init__(self, env='default', **kwargs):
        try:
            self.vis = visdom.Visdom(server="http://localhost", port=8097, use_incoming_socket=False)
            if not self.vis.check_connection():
                print("Visdom server not running. Please run 'python -m visdom.server' first.")
                self.vis = None
        except Exception as e:
            print(f"Could not connect to Visdom server: {e}")
            print("Please run 'python -m visdom.server' first.")
            self.vis = None
        
        self._vis_kw = kwargs
        self.index = {}
        self.log_text = ''

    def _check_connection(self, quiet=False):
        if self.vis is None:
            if not quiet:
                print("No Visdom connection. Please run 'python -m visdom.server' first.")
            return False
        return True

    def reinit(self, env='default', **kwargs):
        """
        change the config of visdom
        """
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        """
        plot multi values
        @params d: dict (name,value) i.e. ('loss',0.11)
        """
        for k, v in d.items():
            if v is not None:
                self.plot(k, v)

    def img_many(self, d):
        for k, v in d.items():
            self.img(k, v)

    def plot(self, name, y, **kwargs):
        """
        self.plot('loss',1.00)
        """
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def img(self, name, img_, **kwargs):
        """
        self.img('input_img',t.Tensor(64,64))
        self.img('input_imgs',t.Tensor(3,64,64))
        self.img('input_imgs',t.Tensor(100,1,64,64))
        self.img('input_imgs',t.Tensor(100,3,64,64),nrows=10)
        !!don't ~~self.img('input_imgs',t.Tensor(100,64,64),nrows=10)~~!!
        """
        self.vis.images(t.Tensor(img_).numpy(),
                        win=name,
                        opts=dict(title=name),
                        **kwargs
                        )

    def log(self, info, win='log_text'):
        """
        self.log({'loss':1,'lr':0.0001})
        """
        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'), \
            info=info))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        return getattr(self.vis, name)

    def state_dict(self):
        return {
            'index': self.index,
            'vis_kw': self._vis_kw,
            'log_text': self.log_text,
            'env': self.vis.env
        }

    def load_state_dict(self, d):
        self.vis = visdom.Visdom(env=d.get('env', self.vis.env), **(self.d.get('vis_kw')))
        self.log_text = d.get('log_text', '')
        self.index = d.get('index', dict())
        return self
    
    def show_points(self, meshes):
        """
        Visualize 3D meshes using Visdom's scatter plot
        Args:
            meshes: list of ModelOutput objects or torch tensors containing vertex coordinates
        """
        if not self._check_connection():
            return

        for i, mesh in enumerate(meshes):
            try:
                # Handle ModelOutput objects
                if hasattr(mesh, 'vertices'):
                    vertices = mesh.vertices
                else:
                    vertices = mesh

                # Ensure vertices is a numpy array
                if t.is_tensor(vertices):
                    vertices = vertices.detach().cpu().numpy()

                # Reshape if needed
                if len(vertices.shape) == 3:  # If vertices has batch dimension
                    vertices = vertices.reshape(-1, 3)

                # Always overwrite same window name
                win_name = 'mesh_view'

                self.vis.scatter(
                    X=vertices,
                    win=win_name,
                    opts=dict(
                        title=f'Mesh View',
                        markersize=2,
                        xlabel='X',
                        ylabel='Y',
                        zlabel='Z',
                        legend=[f'Mesh {i}']
                    )
                )
            except Exception as e:
                print(f"Error visualizing mesh {i}: {e}")
