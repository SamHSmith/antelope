
use glfw::ffi::{glfwCreateWindow, GLFWwindow, GLFWmonitor, glfwGetError};
use std::ptr;
use std::borrow::{Borrow, BorrowMut};
use glfw::{Monitor, Window, WindowEvent, WindowMode, Glfw};
use std::sync::mpsc::Receiver;


pub struct VulkanWindow{
    pub window:Window,
    events : Receiver<(f64, WindowEvent)>
}

impl VulkanWindow{
    pub fn new_windowed(glfw:&Glfw,title:&str ,width:u32,height:u32) -> VulkanWindow{
        VulkanWindow::new_window_internal(glfw,title,width,height,WindowMode::Windowed)
    }

    pub fn new_fullscreen(glfw:&Glfw,title:&str ,width:u32,height:u32,monitor:&Monitor) -> VulkanWindow{
        VulkanWindow::new_window_internal(glfw,title,width,height,WindowMode::FullScreen(monitor))
    }

    pub fn new_windowed_fullscreen(glfw:&Glfw,title:&str ,monitor:&Monitor) -> VulkanWindow{
        let mode=monitor.get_video_mode().expect("Eeeeh, the monitor has no video mode?");

        VulkanWindow::new_fullscreen(glfw,title,mode.width,mode.height,monitor)
    }

    fn new_window_internal(glfw:&Glfw,title:&str ,width:u32,height:u32,mode:WindowMode)-> VulkanWindow{
        unsafe {
            let (window, events)=glfw.create_window(width,height,title,mode).expect("Failed to create a window");
            return VulkanWindow { window, events };
        }
    }
}