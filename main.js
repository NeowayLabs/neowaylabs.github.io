import './style.css';
import changeBackgroundRGB from './changeBackgroundRGB.js';

const body = document.getElementById('bg');

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