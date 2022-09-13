import './style.css';
import changeBackgroundRGB from './changeBackgroundRGB.js';

const body = document.getElementById('bg');

if (typeof DeviceOrientationEvent.requestPermission === 'function') {
  body.addEventListener('click', function () {

    DeviceOrientationEvent
      .requestPermission()
      .then(function() {
        console.log('DeviceOrientationEvent, DeviceMotionEvent enabled');
      })
      .catch(function (error) {
        console.warn('DeviceOrientationEvent, DeviceMotionEvent not enabled', error);
      })

  }, {once: true});
}

if (window.DeviceOrientationEvent) {

  window.addEventListener("deviceorientation", event => {
 
    changeBackgroundRGB({
      element: body,
      red: event.alpha,
      gree: event.gamma,
      blue: event.beta,
    });

  }, true);

}