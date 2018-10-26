"""Select a list of observations from an observation position 
and offset, make cubes/maps (counts, exposure, background) 
and IRFs (EDISP, PSF), and save them to disk.

Goal is to prepare the data for a 3D analysis.
"""
import logging
import numpy as np
import astropy.units as u
from astropy.table import Table, vstack
from regions import CircleSkyRegion
from astropy.coordinates import SkyCoord
from gammapy.data import DataStore
#from gammapy.irf import EnergyDispersion, make_mean_psf, make_mean_edisp
from gammapy.maps import Map, WcsGeom, MapAxis, WcsNDMap
from gammapy.cube import MapMaker, MapEvaluator, MapFit, PSFKernel

log = logging.getLogger()


class Config:
    target_glon = 359.944 * u.deg  #actual SgrA* position
    target_glat = -0.046 * u.deg
    # target_radius = 0.5 * u.deg
    target_skycoord = SkyCoord(target_glon, target_glat, frame="galactic")

    on_radius = 1 * u.deg
    on_region = CircleSkyRegion(target_skycoord, on_radius)

    offset_max = 10 * u.deg

    # energy_min = 0.1 * u.TeV
    # energy_max = 160 * u.TeV
    # energy_nbins = 35
    # energy_bins = np.logspace(
    #     start=np.log10(energy_min.value),
    #     stop=np.log10(energy_max.value),
    #     num=energy_nbins,
    # )

    energy_axis = MapAxis.from_bounds(
        0.1, 160, 35, unit="TeV", name="energy", interp="log"
    )
    map_geom = WcsGeom.create(
        skydir=target_skycoord,
        width=(8, 8),
        binsz=0.02,
        coordsys="GAL",
        axes=[energy_axis],
    )

    # Manually from header keywords;
    # Could write a helper function to extract that automatically
    # and also total number of counts for each component
    mc_id = {1: "bkg", 2: "iem", 44: "j1702", 110: "pwn_8"}

    # Options: 'iem', 'j1702', 'pwn_8', 'all'
    model = "all"
    add_background = True

    debug = False


config = Config()



def get_observations():
    data_store = DataStore.from_dir("$CTADATA/index/gc")
    t = data_store.obs_table
    pos_obs = SkyCoord(t["RA_PNT"], t["DEC_PNT"], unit="deg")
    offset = config.target_skycoord.separation(pos_obs)
    mask = offset < config.offset_max
    obs_id = t["OBS_ID"][mask]
    if config.debug:
        obs_id = obs_id[:3]
    return data_store.obs_list(obs_id)

def make_data_maps():
    log.info("Executing make_data_maps")
    observations = get_observations()
    maker = MapMaker(geom=config.map_geom, offset_max=config.offset_max)
    maker.run(observations)

    for name, map in maker.maps.items():
        filename = f"{name}_cube.fits.gz"
        log.info(f"Writing {filename}")
        map.write(filename, overwrite=True)

    for name, image in maker.make_images().items():
        filename = f"{name}_image.fits.gz"
        log.info(f"Writing {filename}")
        image.write(filename, overwrite=True)


def make_irfs():
	log.info("Executing make_irfs")
	observations = get_observations()
	#mean PSF	
	src_pos = config.target_skycoord
	table_psf = observations.make_mean_psf(src_pos)

	# PSF kernel used for the model convolution
	psf_kernel = PSFKernel.from_table_psf(table_psf, geom=config.map_geom, max_radius="0.5 deg")

	# define energy grid
	energy = config.energy_axis.edges * config.energy_axis.unit

	# mean edisp
	edisp = observations.make_mean_edisp(position=src_pos, e_true=energy, e_reco=energy)

	log.info("Writing psf.fits.gz and edisp.fits.gz")
	psf_kernel.write("psf.fits.gz",overwrite=True)
	edisp.write("edisp.fits.gz",overwrite=True)


def main():
  
    make_data_maps()
    make_irfs()  


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    main()

