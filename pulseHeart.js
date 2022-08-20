import {html, css, LitElement} from 'lit';

export class PulseHeart extends LitElement {
  
  static styles = css`
    :host {
        --color: pink;
        --size: 200px;
        display: inline-block;
        margin: 0 14px;
    }

    .pulse-heart {
        background: var(--color);
        width: var(--size);
        height: var(--size);
        position: relative;
        animation: pulse 1s infinite alternate;
    }
    
    @keyframes pulse {
        0%   {transform: rotate(-135deg) scale(0.8);}
        100% {transform: rotate(-135deg) scale(1.0);}
    }
    
    .pulse-heart:after {
        content: "";
        display: block;
        width: var(--size);
        height: var(--size);
        background: var(--color);
        position: absolute;
        left: 50%;
        border-radius: 50%;
    }
    
    .pulse-heart:before {
        content: "";
        display: block;
        width: var(--size);
        height: var(--size);
        background: var(--color);
        position: absolute;
        top: 50%;
        border-radius: 50%;
    }
  `;

  render() {
    return html`<div class=pulse-heart></div>`;
  }

}

customElements.define('pulse-heart', PulseHeart);