"""
Describe purpose of this script here

Created: 1/28/25
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as img

from spiceypy import furnsh, str2et, etcal, spkezr, timout

from rocket_sim.planet import Earth
from voyager.voyager_depart_target import init_spice


def main():
    init_spice()
    earth=Earth()
    earthmap=img.imread('data/EarthMap.jpg')
    plt.figure("Earth map")
    plt.imshow(earthmap,extent=[-180,180,-90,90])
    plt.axis('equal')
    plt.pause(0.1)
    plt.xlim(-180,180)
    plt.ylim(-90,90)
    for vgr_id in (1,2):
        furnsh(f"data/vgr{vgr_id}_st.bsp")
        furnsh(f"products/vgr{vgr_id}_pm.bsp")
        furnsh(f"products/vgr{vgr_id}_centaur2.bsp")
        cal0={2:"1977-08-20 14:40:18.533 TDB",1:"1977-09-05 13:06:45.234"}[vgr_id]
        et0=str2et(cal0)
        print(et0,etcal(et0))
        dt=2*60*60
        et1=et0+dt
        lats=[]
        lons=[]
        with open(Path("products")/f"vgr{vgr_id}_depart.kml","wt") as ouf:
            print(
fr'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2" xmlns:kml="http://www.opengis.net/kml/2.2" xmlns:atom="http://www.w3.org/2005/Atom">
<Document>
	<name>Voyager {vgr_id} Depart</name>
	<Style id="multiTrack_n">
		<IconStyle>
			<Icon>
				<href>http://earth.google.com/images/kml-icons/track-directional/track-0.png</href>
			</Icon>
		</IconStyle>
		<LineStyle>
			<color>99{"0000ff" if vgr_id==2 else "00ff00"}</color>
			<width>1</width>
		</LineStyle>
	</Style>
	<Style id="multiTrack_h">
		<IconStyle>
			<scale>1.2</scale>
			<Icon>
				<href>http://earth.google.com/images/kml-icons/track-directional/track-0.png</href>
			</Icon>
		</IconStyle>
		<LineStyle>
			<color>99{"0000ff" if vgr_id==2 else "00ff00"}</color>
			<width>8</width>
		</LineStyle>
	</Style>
	<StyleMap id="multiTrack">
		<Pair>
			<key>normal</key>
			<styleUrl>#multiTrack_n</styleUrl>
		</Pair>
		<Pair>
			<key>highlight</key>
			<styleUrl>#multiTrack_h</styleUrl>
		</Pair>
	</StyleMap>
	<Placemark>
		<name>Voyager {vgr_id}</name>
		<styleUrl>#multiTrack</styleUrl>
		<gx:balloonVisibility>1</gx:balloonVisibility>
		<gx:Track>
            <altitudeMode>absolute</altitudeMode>''',file=ouf)

            for i in range(dt):
                et=et0+i
                state,lt=spkezr(f"{-30-vgr_id}",et,"IAU_EARTH","NONE","399")
                r=state[:3].reshape(-1,1)*1000
                lla=earth.b2lla(r,deg=True)
                lats.append(lla.lat)
                lons.append(lla.lon)
                print(f'    <when>{timout(et,"YYYY-MM-DDTHR:MN:SC.###Z ::UTC",25)}</when>'
                      f'<gx:coord>{lla.lon} {lla.lat} {lla.alt}</gx:coord>',file=ouf)
            print(
r'''
        </gx:Track>
    </Placemark>
</Document>

</kml>''',file=ouf)
        print(f'Voyager {vgr_id}: lat0={lats[0]}deg, lon0={lons[0]}deg')
        plt.plot(lons,lats,'r-' if vgr_id==2 else 'g-')
    plt.show()

if __name__ == "__main__":
    main()
