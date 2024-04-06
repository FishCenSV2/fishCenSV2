#include <iostream>
#include "session.hpp"

using boost::asio::ip::tcp;

/** @brief Constructor for Session.
*
* @param socket A socket object (note object is moved).
* @return None
*/
Session::Session(tcp::socket socket, Server& server)
    : _socket(std::move(socket)), server(server) {
}

/** @brief Starts the session by calling do_read()
*
* @return None
*/
void Session::start() {
    do_read();
}

/** @brief Reads data from the client
*
* @return None
*/
void Session::do_read() {

    //Get shared pointer to Session object
    auto self(shared_from_this());

    boost::asio::async_read(_socket, boost::asio::buffer(&request, sizeof(request)),
        [this, self](boost::system::error_code ec, std::size_t len) {
            if (!ec) {

                do_write();
            }

            else {
                std::cout << ec.message() << "\n";
            }
        });
   
}

/** @brief Writes image frame to client
*
* @return None
*/
void Session::do_write() {

    //Get shared pointer to Session object
    auto self(shared_from_this());

    {
        std::lock_guard<std::mutex> guard(server.mutex);

        /*
        The thread function in main modifies the buffer in a way that invalidates all the
        iterators for all references. So we resort to a copy to a member variable. This must
        be a member variable since boost::asio::buffer does not own the object passed in to it
        meaning we must guarantee its lifetime ourselves.
        */
        data_cpy = server.data;
    }

    boost::asio::async_write(_socket, boost::asio::buffer(data_cpy),
        [this, self](boost::system::error_code ec, std::size_t len) {
            if (!ec) {
                do_read();
            }

            else {
                std::cout << ec.message() << "\n";
            }

        });
}   