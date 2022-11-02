import os
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_domain_s(lat_path, arr1):
    
    if len(arr1.shape) == 3:
        arr1 = arr1[:,:,0]
    
    
    LAT = np.load(os.path.join(lat_path,"lat.npy"))
    LON = np.load(os.path.join(lat_path,"lon.npy"))

    lat1 = LAT[38:70,81:113]
    lon1 = LON[38:70,81:113]
    fig = plt.figure(figsize=(15, 10)) #Define size of figure
    title = "Imputed" 
    # fig.suptitle(title,y=0.95,fontsize=20,fontweight="bold")
    cmap = plt.get_cmap('Set2', 3)

    # ax = plt.axes(projection=ccrs.PlateCarree())
    ax = plt.subplot(2, 2, 1, projection=ccrs.PlateCarree())

    ax.set_extent([125.5, 132.5, 33, 40],crs=ccrs.PlateCarree())
    ax.set_title('Batch_real', fontsize=15)
    cm=ax.pcolormesh(lon1,lat1,arr1,cmap="Greys",vmin=0, vmax=1, transform=ccrs.PlateCarree())
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS)


    cbar_ax = fig.add_axes([0.83, 0.13, 0.02, 0.67])
    cbar = fig.colorbar(cm, cax=cbar_ax, pad=0.1)
    cbar.set_label('Index of Agreement', labelpad=10, fontsize=17)
    
    return fig 


def plot_domain(lat_path, arr1, arr2, arr3, ioa):
    
    
    if len(arr1.shape) == 3:
        arr1 = arr1[:,:,0]
    if len(arr2.shape) == 3:
        arr2 = arr2[:,:,0]
    
    if len(arr3.shape) == 3:
        arr3 = arr3[:,:,0]
    
    LAT = np.load(os.path.join(lat_path,"lat.npy"))
    LON = np.load(os.path.join(lat_path,"lon.npy"))

    lat1 = LAT[38:70,81:113]
    lon1 = LON[38:70,81:113]
    fig = plt.figure(figsize=(15, 10)) #Define size of figure
    title = "Imputed" 
    # fig.suptitle(title,y=0.95,fontsize=20,fontweight="bold")
    cmap = plt.get_cmap('Set2', 3)

    # ax = plt.axes(projection=ccrs.PlateCarree())
    ax = plt.subplot(2, 2, 1, projection=ccrs.PlateCarree())

    ax.set_extent([125.5, 132.5, 33, 40],crs=ccrs.PlateCarree())
    ax.set_title('Batch_real', fontsize=15)
    cm=ax.pcolormesh(lon1,lat1,arr1,cmap="YlOrRd",vmin=0, vmax=arr1.max(), transform=ccrs.PlateCarree())
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS)
    fig.colorbar(cm, ax=ax, fraction=0.046, pad=0.04) 
    
    
    arr2[arr2==1]=np.nan
    ax2 = plt.subplot(2, 2, 2, projection=ccrs.PlateCarree())
    ax2.set_extent([125.5, 132.5, 33, 40],crs=ccrs.PlateCarree())
    ax2.set_title('Masked Input', fontsize=15)
    cm=ax2.pcolormesh(lon1,lat1,arr2,cmap="YlOrRd",vmin=0, vmax=arr1.max(), transform=ccrs.PlateCarree())
    ax2.coastlines(resolution='10m')
    ax2.add_feature(cfeature.BORDERS)
    fig.colorbar(cm, ax=ax2, fraction=0.046, pad=0.04) 
    
    ax3 = plt.subplot(2, 2, 3, projection=ccrs.PlateCarree())
    ax3.set_extent([125.5, 132.5, 33, 40],crs=ccrs.PlateCarree())
    ax3.set_title('Imputed', fontsize=15)
    cm=ax3.pcolormesh(lon1,lat1,arr3,cmap="YlOrRd",vmin=0, vmax=arr1.max(), transform=ccrs.PlateCarree())
    ax3.coastlines(resolution='10m')
    ax3.add_feature(cfeature.BORDERS)
    fig.colorbar(cm, ax=ax3, fraction=0.046, pad=0.04) 
    
    ax4 = plt.subplot(2, 2, 4, projection=ccrs.PlateCarree())
    ax4.set_extent([125.5, 132.5, 33, 40],crs=ccrs.PlateCarree())
    ax4.set_title('Bias P-O', fontsize=15)
    cm=ax4.pcolormesh(lon1,lat1,arr1-arr3,cmap='Reds_r',vmin=(arr3 - arr1).min(), vmax=(arr1-arr3).max(), transform=ccrs.PlateCarree())
    ax4.coastlines(resolution='10m')
    ax4.add_feature(cfeature.BORDERS)
    fig.colorbar(cm, ax=ax4, fraction=0.046, pad=0.04) 
    

    

    fig.suptitle(f"IOA : {ioa:.3f}") 
    
    return fig


def plot_domain_val(lat_path, arr1, arr2, arr3, ioa, full_ioa):
    
    
    if len(arr1.shape) == 3:
        arr1 = arr1[:,:,0]
    if len(arr2.shape) == 3:
        arr2 = arr2[:,:,0]
    
    if len(arr3.shape) == 3:
        arr3 = arr3[:,:,0]
    
    LAT = np.load(os.path.join(lat_path,"lat.npy"))
    LON = np.load(os.path.join(lat_path,"lon.npy"))

    lat1 = LAT[38:70,81:113]
    lon1 = LON[38:70,81:113]
    fig = plt.figure(figsize=(15, 10)) #Define size of figure
    title = "Imputed" 
    # fig.suptitle(title,y=0.95,fontsize=20,fontweight="bold")
    cmap = plt.get_cmap('Set2', 3)

    # ax = plt.axes(projection=ccrs.PlateCarree())
    ax = plt.subplot(2, 2, 1, projection=ccrs.PlateCarree())

    ax.set_extent([125.5, 132.5, 33, 40],crs=ccrs.PlateCarree())
    ax.set_title('Batch_real', fontsize=15)
    cm=ax.pcolormesh(lon1,lat1,arr1,cmap="YlOrRd",vmin=0, vmax=arr1.max(), transform=ccrs.PlateCarree())
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS)
    fig.colorbar(cm, ax=ax, fraction=0.046, pad=0.04) 
    
    
    arr2[arr2==1]=np.nan
    ax2 = plt.subplot(2, 2, 2, projection=ccrs.PlateCarree())
    ax2.set_extent([125.5, 132.5, 33, 40],crs=ccrs.PlateCarree())
    ax2.set_title('Masked Input', fontsize=15)
    cm=ax2.pcolormesh(lon1,lat1,arr2,cmap="YlOrRd",vmin=0, vmax=arr1.max(), transform=ccrs.PlateCarree())
    ax2.coastlines(resolution='10m')
    ax2.add_feature(cfeature.BORDERS)
    fig.colorbar(cm, ax=ax2, fraction=0.046, pad=0.04) 
    
    ax3 = plt.subplot(2, 2, 3, projection=ccrs.PlateCarree())
    ax3.set_extent([125.5, 132.5, 33, 40],crs=ccrs.PlateCarree())
    ax3.set_title('Imputed', fontsize=15)
    cm=ax3.pcolormesh(lon1,lat1,arr3,cmap="YlOrRd",vmin=0, vmax=arr1.max(), transform=ccrs.PlateCarree())
    ax3.coastlines(resolution='10m')
    ax3.add_feature(cfeature.BORDERS)
    fig.colorbar(cm, ax=ax3, fraction=0.046, pad=0.04) 
    
    ax4 = plt.subplot(2, 2, 4, projection=ccrs.PlateCarree())
    ax4.set_extent([125.5, 132.5, 33, 40],crs=ccrs.PlateCarree())
    ax4.set_title('Bias P-O', fontsize=15)
    cm=ax4.pcolormesh(lon1,lat1,arr1-arr3,cmap='Reds_r',vmin=(arr3 - arr1).min(), vmax=(arr1-arr3).max(), transform=ccrs.PlateCarree())
    ax4.coastlines(resolution='10m')
    ax4.add_feature(cfeature.BORDERS)
    fig.colorbar(cm, ax=ax4, fraction=0.046, pad=0.04) 
    

    

    fig.suptitle(f"IOA : {ioa:.3f}, FULL IOA: {full_ioa:.3f}") 
    
    return fig